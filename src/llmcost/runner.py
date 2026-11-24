from __future__ import annotations

import asyncio
import csv
import json
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from .backends import GenResult, make_backend
from .checkers import check
from .config import Config, ModelCfg, Prices
from .energy import PowerSampler, make_sampler
from .tasks import Task

CSV_FIELDS = [
    "run_id", "timestamp_utc", "model_name", "backend", "model_id", "task_set", "task_id", "repeat",
    "correct", "error", "stop_reason",
    "input_tokens", "output_tokens", "cache_write_tokens", "cache_read_tokens",
    "latency_s", "gen_tokens_per_s", "prompt_tokens_per_s", "load_s",
    "energy_wh_gross", "energy_wh_net", "avg_power_w", "idle_power_w", "energy_coverage",
    "cost_usd", "cost_usd_batch", "price_basis",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def host_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "packages": {p: version(p) for p in ("anthropic", "ollama")},
    }
    if platform.system() == "Darwin":
        info |= {
            "hw_model": _sh("sysctl", "-n", "hw.model"),
            "cpu": _sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "memsize_bytes": _sh("sysctl", "-n", "hw.memsize"),
            "on_ac_power": "AC Power" in _sh("pmset", "-g", "batt"),
        }
    elif platform.system() == "Windows":
        info |= {
            "cpu": _sh("powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"),
            "board": _sh("powershell", "-NoProfile", "-Command",
                         "$c=Get-CimInstance Win32_ComputerSystem; \"$($c.Manufacturer) $($c.Model)\""),
            "memsize_bytes": _sh("powershell", "-NoProfile", "-Command",
                                 "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"),
        }
    if shutil.which("nvidia-smi"):
        info["gpu"] = _sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version,power.limit",
                          "--format=csv,noheader")
    if shutil.which("ollama"):
        info["ollama_version"] = _sh("ollama", "--version")
    return info


class ResultWriter:
    def __init__(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = out_dir / "results.csv"
        self.raw_path = out_dir / "raw.jsonl"
        self.runs_path = out_dir / "runs.jsonl"
        self._new = not self.csv_path.exists()

    def done_keys(self) -> set[tuple[str, str, int]]:
        if self._new:
            return set()
        with self.csv_path.open(newline="", encoding="utf-8") as f:
            return {(r["model_name"], r["task_id"], int(r["repeat"])) for r in csv.DictReader(f) if not r["error"]}

    def write(self, row: dict[str, Any], raw: dict[str, Any]) -> None:
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
            if self._new:
                w.writeheader()
                self._new = False
            w.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in CSV_FIELDS})
        with self.raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(raw, ensure_ascii=False) + "\n")

    def write_run(self, meta: dict[str, Any]) -> None:
        with self.runs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")


def _rows(
    run_id: str, m: ModelCfg, task: Task, repeat: int, g: GenResult, correct: bool | None,
    extracted: str, detail: str, energy: dict[str, float | None], cost: tuple[float | None, float | None, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    row = {
        "run_id": run_id, "timestamp_utc": _now(), "model_name": m.name, "backend": m.backend,
        "model_id": m.model, "task_set": task.task_set, "task_id": task.task_id, "repeat": repeat,
        "correct": "" if correct is None else int(correct), "error": g.error, "stop_reason": g.stop_reason,
        "input_tokens": g.input_tokens, "output_tokens": g.output_tokens,
        "cache_write_tokens": g.cache_write_tokens, "cache_read_tokens": g.cache_read_tokens,
        "latency_s": round(g.latency_s, 4),
        "gen_tokens_per_s": None if g.gen_tokens_per_s is None else round(g.gen_tokens_per_s, 3),
        "prompt_tokens_per_s": None if g.prompt_tokens_per_s is None else round(g.prompt_tokens_per_s, 3),
        "load_s": g.load_s, **energy,
        "cost_usd": cost[0], "cost_usd_batch": cost[1], "price_basis": cost[2],
    }
    raw = {"run_id": run_id, "model_name": m.name, "task_id": task.task_id, "repeat": repeat,
           "solution": extracted, "check_detail": detail, "output": g.text}
    return row, raw


async def run_api(m: ModelCfg, tasks: list[Task], repeats: int, cfg: Config, prices: Prices, w: ResultWriter) -> None:
    if m.model not in prices.models:
        raise KeyError(f"нет цены для {m.model} в prices.toml")
    backend = make_backend(m, cfg.run)
    run_id = f"{m.name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    done = w.done_keys()
    todo = [(t, r) for r in range(1, repeats + 1) for t in tasks if (m.name, t.task_id, r) not in done]
    print(f"[{m.name}] {len(todo)} задач (пропущено готовых: {len(tasks) * repeats - len(todo)})")
    if not todo:
        await backend.aclose()
        return
    sem = asyncio.Semaphore(cfg.run.api_concurrency)
    started = _now()

    async def one(task: Task, rep: int) -> tuple[Task, int, GenResult]:
        async with sem:
            return task, rep, await backend.generate(task.prompt)

    n_ok = n = 0
    try:
        for fut in asyncio.as_completed([one(t, r) for t, r in todo]):
            task, rep, g = await fut
            correct, extracted, detail, cost = None, "", "", (None, None, f"api {prices.fixed_on}")
            if not g.error:
                res = await asyncio.to_thread(check, task, g.text, cfg.run.code_timeout_s)
                correct, extracted, detail = res.correct, res.extracted, res.detail
                std, batch = prices.cost_usd(m.model, g.input_tokens, g.output_tokens,
                                             g.cache_write_tokens, g.cache_read_tokens)
                cost = (round(std, 8), round(batch, 8), f"api {prices.fixed_on}")
                n += 1
                n_ok += int(correct)
            empty = {"energy_wh_gross": None, "energy_wh_net": None, "avg_power_w": None,
                     "idle_power_w": None, "energy_coverage": None}
            w.write(*_rows(run_id, m, task, rep, g, correct, extracted, detail, empty, cost))
            print(f"  {task.task_id} r{rep}: {'OK ' if correct else 'ERR' if g.error else 'no '} "
                  f"out={g.output_tokens} {g.latency_s:.1f}s  [{n_ok}/{n}]")
    finally:
        await backend.aclose()
        w.write_run({"run_id": run_id, "model": asdict(m), "started": started, "finished": _now(),
                     "n_tasks": len(todo), "prices_fixed_on": prices.fixed_on,
                     "backend_info": await backend.info(), "host": host_info(), "run_cfg": asdict(cfg.run)})


async def run_local(m: ModelCfg, tasks: list[Task], repeats: int, cfg: Config, w: ResultWriter, energy: bool) -> None:
    backend = make_backend(m, cfg.run)
    run_id = f"{m.name}-{datetime.now(UTC):%Y%m%dT%H%M%SZ}"
    done = w.done_keys()
    todo = [(t, r) for r in range(1, repeats + 1) for t in tasks if (m.name, t.task_id, r) not in done]
    print(f"[{m.name}] {len(todo)} задач (пропущено готовых: {len(tasks) * repeats - len(todo)})")
    if not todo:
        return
    p_el = cfg.economics.base_electricity

    print(f"[{m.name}] прогрев: загрузка весов в память")
    warm = await backend.generate("Reply with the single word OK.")
    if warm.error:
        raise RuntimeError(f"Ollama не отвечает: {warm.error}")
    info = await backend.info()

    sampler: PowerSampler | None = None
    idle_w: float | None = None
    if energy:
        sampler = make_sampler()
        await sampler.start()
        print(f"[{m.name}] простой {cfg.run.idle_seconds:.0f} с: не трогайте компьютер")
        idle_w = await sampler.idle_watts(cfg.run.idle_seconds)
        print(f"[{m.name}] мощность в простое ({sampler.source}): {idle_w:.2f} W")

    started, t_start = _now(), time.monotonic()
    print(f"START {started}  <- сбросьте/запишите показания ваттметра")
    gen_seconds = 0.0
    n_ok = n = 0
    try:
        for task, rep in todo:
            a = time.monotonic()
            g = await backend.generate(task.prompt)
            b = time.monotonic()
            gen_seconds += b - a
            e: dict[str, float | None] = {"energy_wh_gross": None, "energy_wh_net": None, "avg_power_w": None,
                                          "idle_power_w": idle_w, "energy_coverage": None}
            if sampler is not None and idle_w is not None:
                joules, covered = await sampler.energy_j(a, b)
                e |= {"energy_wh_gross": round(joules / 3600, 6),
                      "energy_wh_net": round((joules - idle_w * (b - a)) / 3600, 6),
                      "avg_power_w": round(joules / (b - a), 3),
                      "energy_coverage": round(covered / (b - a), 3)}
            correct, extracted, detail, cost = None, "", "", (None, None, "")
            if not g.error:
                res = await asyncio.to_thread(check, task, g.text, cfg.run.code_timeout_s)
                correct, extracted, detail = res.correct, res.extracted, res.detail
                n += 1
                n_ok += int(correct)
                if e["energy_wh_net"] is not None:
                    cost = (round(e["energy_wh_net"] / 1000 * p_el, 8), None, f"electricity {p_el} USD/kWh")
            w.write(*_rows(run_id, m, task, rep, g, correct, extracted, detail, e, cost))
            tps = f"{g.gen_tokens_per_s:.1f} tok/s" if g.gen_tokens_per_s else ""
            pw = f"{e['avg_power_w']:.1f} W" if e["avg_power_w"] else ""
            print(f"  {task.task_id} r{rep}: {'OK ' if correct else 'ERR' if g.error else 'no '} "
                  f"out={g.output_tokens} {g.latency_s:.1f}s {tps} {pw}  [{n_ok}/{n}]")
    finally:
        finished, duration = _now(), time.monotonic() - t_start
        print(f"END {finished}  длительность {duration:.0f} с  <- запишите показания ваттметра в results/wall_meter.csv")
        if sampler:
            await sampler.stop()
        w.write_run({"run_id": run_id, "model": asdict(m), "started": started, "finished": finished,
                     "duration_s": round(duration, 2), "generation_s": round(gen_seconds, 2),
                     "n_tasks": len(todo), "idle_power_w": idle_w,
                     "energy_sensor": sampler.source if sampler else None,
                     "sensor_parse_failures": sampler.parse_failures if sampler else None,
                     "backend_info": info, "host": host_info(), "run_cfg": asdict(cfg.run),
                     "hardware_cfg": cfg.hardware})
