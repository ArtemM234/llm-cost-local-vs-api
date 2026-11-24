"""Aggregates results.csv → summary.csv (per model) and breakeven.csv (V* grid)."""

from __future__ import annotations

import csv
import json
import math
import statistics as st
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Config
from .costmodel import break_even_volume, max_daily_volume, owned_cost_per_correct, wilson_interval


@dataclass(slots=True)
class ModelSummary:
    model_name: str
    backend: str
    task_set: str
    n: int
    n_correct: int
    acc: float
    acc_ci_lo: float
    acc_ci_hi: float
    t_task_s: float
    out_tokens_mean: float
    gen_tps_median: float | None
    cost_usd_mean: float | None  # API only
    cost_usd_batch_mean: float | None
    energy_wh_net_mean: float | None  # sensor (powermetrics SoC / nvidia-smi GPU), minus idle
    energy_wh_net_sd: float | None
    energy_wh_wall_mean: float | None  # wall meter, whole machine (if recorded)
    n_errors: int


def _f(x: str) -> float | None:
    return float(x) if x not in ("", None) else None


def _mean(xs: list[float]) -> float | None:
    return st.fmean(xs) if xs else None


def load_rows(results_dir: Path) -> list[dict[str, str]]:
    with (results_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def wall_energy_per_task(results_dir: Path) -> dict[str, float]:
    """wall_meter.csv: run_id,wh_measured,idle_w — meter reading between START and END lines,
    and the machine's idle wall power. Returns Wh per task per model_name (task-weighted)."""
    meter, runs = results_dir / "wall_meter.csv", results_dir / "runs.jsonl"
    if not meter.exists() or not runs.exists():
        return {}
    meta = {r["run_id"]: r for r in map(json.loads, runs.read_text(encoding="utf-8").splitlines()) if r}
    total: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    with meter.open(newline="", encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            run = meta.get(rec["run_id"])
            if not run or not run.get("n_tasks"):
                continue
            net_wh = float(rec["wh_measured"]) - float(rec["idle_w"]) * run["duration_s"] / 3600
            acc = total[run["model"]["name"]]
            acc[0] += net_wh
            acc[1] += run["n_tasks"]
    return {k: wh / n for k, (wh, n) in total.items() if n}


def summarize(rows: list[dict[str, str]], wall: dict[str, float]) -> list[ModelSummary]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[(r["model_name"], r["task_set"])].append(r)
    out = []
    for (name, task_set), rs in sorted(groups.items()):
        ok = [r for r in rs if not r["error"]]
        # a task that later succeeded on resume must not count its failed attempt as an error
        done_keys = {(r["task_id"], r["repeat"]) for r in ok}
        errors = {(r["task_id"], r["repeat"]) for r in rs if r["error"]} - done_keys
        k = sum(int(r["correct"]) for r in ok)
        lo, hi = wilson_interval(k, len(ok))
        energy = [v for r in ok if (v := _f(r["energy_wh_net"])) is not None]
        tps = [v for r in ok if (v := _f(r["gen_tokens_per_s"])) is not None]
        out.append(ModelSummary(
            model_name=name, backend=rs[0]["backend"], task_set=task_set, n=len(ok), n_correct=k,
            acc=k / len(ok) if ok else math.nan, acc_ci_lo=lo, acc_ci_hi=hi,
            t_task_s=st.fmean(float(r["latency_s"]) for r in ok) if ok else math.nan,
            out_tokens_mean=st.fmean(int(r["output_tokens"]) for r in ok) if ok else math.nan,
            gen_tps_median=st.median(tps) if tps else None,
            cost_usd_mean=_mean([v for r in ok if r["backend"] == "anthropic" and (v := _f(r["cost_usd"])) is not None]),
            cost_usd_batch_mean=_mean([v for r in ok if (v := _f(r["cost_usd_batch"])) is not None]),
            energy_wh_net_mean=_mean(energy),
            energy_wh_net_sd=st.stdev(energy) if len(energy) > 1 else None,
            energy_wh_wall_mean=wall.get(name),
            n_errors=len(errors),
        ))
    return out


def breakeven_grid(summaries: list[ModelSummary], cfg: Config) -> list[dict[str, object]]:
    eco = cfg.economics
    rows: list[dict[str, object]] = []
    by_set: dict[str, list[ModelSummary]] = defaultdict(list)
    for s in summaries:
        by_set[s.task_set].append(s)
    for task_set, ss in by_set.items():
        for loc in (s for s in ss if s.backend == "ollama"):
            # wall meter covers the whole machine → preferred when available
            e_wh, e_src = ((loc.energy_wh_wall_mean, "wall") if loc.energy_wh_wall_mean is not None
                           else (loc.energy_wh_net_mean, "sensor"))
            if e_wh is None:
                continue
            e_kwh = e_wh / 1000
            v_max = max_daily_volume(loc.t_task_s)
            for api in (s for s in ss if s.backend == "anthropic" and s.cost_usd_mean is not None):
                for pricing, c_api in (("standard", api.cost_usd_mean), ("batch", api.cost_usd_batch_mean)):
                    if c_api is None or not api.acc:
                        continue
                    api_per_correct = c_api / api.acc
                    for p_el in eco.electricity_usd_per_kwh:
                        owned = owned_cost_per_correct(e_kwh, p_el, loc.t_task_s, eco.wear_usd_per_hour, loc.acc) if loc.acc else math.inf
                        for life in eco.lifetimes_years:
                            v_star = break_even_volume(eco.laptop_price_usd, life, e_kwh, p_el, loc.acc, c_api, api.acc)
                            rows.append({
                                "task_set": task_set, "local": loc.model_name, "api": api.model_name,
                                "api_pricing": pricing, "laptop_usd": eco.laptop_price_usd, "lifetime_years": life,
                                "electricity_usd_kwh": p_el, "energy_source": e_src,
                                "acc_local": round(loc.acc, 4), "acc_api": round(api.acc, 4),
                                "api_usd_per_correct": api_per_correct,
                                "owned_local_usd_per_correct": owned,
                                "owned_local_cheaper": owned < api_per_correct,
                                "v_star_per_day": v_star, "v_max_per_day": round(v_max, 1),
                                "feasible": v_star <= v_max,
                            })
    return rows


def bandwidth_check(results_dir: Path, summaries: list[ModelSummary], cfg: Config) -> list[str]:
    """tokens/s ≈ bandwidth / weights size: measured decode speed vs the memory-bound ceiling."""
    bw = float(cfg.hardware.get("memory_bandwidth_gbs") or 0)
    runs_path = results_dir / "runs.jsonl"
    if not bw or not runs_path.exists():
        return ["(пропущено: укажите hardware.memory_bandwidth_gbs в config.toml)"]
    weights: dict[str, int] = {}
    for r in map(json.loads, filter(None, runs_path.read_text(encoding="utf-8").splitlines())):
        if (b := r.get("backend_info", {}).get("weights_bytes")):
            weights[r["model"]["name"]] = b
    lines = []
    for s in summaries:
        if s.backend == "ollama" and s.model_name in weights and s.gen_tps_median:
            ceiling = bw * 1e9 / weights[s.model_name]
            lines.append(f"{s.model_name}: веса {weights[s.model_name] / 1e9:.2f} GB, потолок {ceiling:.1f} tok/s, "
                         f"измерено {s.gen_tps_median:.1f} tok/s ({s.gen_tps_median / ceiling:.0%} от потолка)")
    return lines


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def analyze(results_dir: Path, cfg: Config) -> None:
    rows = load_rows(results_dir)
    summaries = summarize(rows, wall_energy_per_task(results_dir))
    write_csv(results_dir / "summary.csv", [asdict(s) for s in summaries])
    grid = breakeven_grid(summaries, cfg)
    write_csv(results_dir / "breakeven.csv", grid)

    print(f"{'модель':<22}{'набор':<16}{'n':>5}{'acc':>7}{'95% CI':>15}{'t, с':>8}{'$/задача':>11}{'Wh/задача':>11}")
    for s in summaries:
        ci = f"{s.acc_ci_lo:.2f}–{s.acc_ci_hi:.2f}"
        cost = f"{s.cost_usd_mean:.6f}" if s.cost_usd_mean is not None else "—"
        wh = s.energy_wh_wall_mean if s.energy_wh_wall_mean is not None else s.energy_wh_net_mean
        print(f"{s.model_name:<22}{s.task_set:<16}{s.n:>5}{s.acc:>7.3f}{ci:>15}{s.t_task_s:>8.1f}{cost:>11}"
              f"{(f'{wh:.4f}' if wh is not None else '—'):>11}" + (f"  ошибок: {s.n_errors}" if s.n_errors else ""))

    eco = cfg.economics
    base = [r for r in grid if r["electricity_usd_kwh"] == eco.base_electricity and r["api_pricing"] == "standard"
            and r["lifetime_years"] == eco.lifetimes_years[len(eco.lifetimes_years) // 2]]
    if base:
        print(f"\nV* (покупка компьютера ${eco.laptop_price_usd:.0f}, срок {base[0]['lifetime_years']} г., "
              f"{eco.base_electricity} $/кВт·ч, обычные цены API):")
        for r in base:
            v = "никогда" if math.isinf(r["v_star_per_day"]) else f"{r['v_star_per_day']:.0f}/день"
            note = "" if r["feasible"] else " (> V_max: физически недостижимо)"
            print(f"  {r['local']} vs {r['api']} [{r['task_set']}]: V* = {v}{note}, V_max = {r['v_max_per_day']:.0f}/день; "
                  f"компьютер уже есть: локально {r['owned_local_usd_per_correct']:.6f} vs API {r['api_usd_per_correct']:.6f} $/верную")
    print("\nПропускная способность памяти:")
    for line in bandwidth_check(results_dir, summaries, cfg):
        print("  " + line)
    print(f"\nЗаписано: {results_dir / 'summary.csv'}, {results_dir / 'breakeven.csv'}")
