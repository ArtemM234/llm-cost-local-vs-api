from __future__ import annotations

import csv
import math
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from llmcost.analyze import load_rows, summarize, wall_energy_per_task
from llmcost.config import load_config
from llmcost.costmodel import break_even_volume, local_cost_per_correct, max_daily_volume

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"

LOCAL, SONNET, HAIKU = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2,
})

money = FuncFormatter(lambda v, _: f"${v:g}")
count = FuncFormatter(lambda v, _: f"{v:,.0f}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config.toml")
    eco = cfg.economics
    rows = load_rows(ROOT / "results")
    s = {x.model_name: x for x in summarize(rows, wall_energy_per_task(ROOT / "results"))}
    loc, son, hai = s["qwen3-8b"], s["sonnet-5.5"], s["haiku-5.5"]
    e_kwh = loc.energy_wh_net_mean / 1000
    p_el = eco.base_electricity
    life = eco.lifetimes_years[len(eco.lifetimes_years) // 2]
    c_hw = eco.laptop_price_usd
    v_max = max_daily_volume(loc.t_task_s)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    vols = [10 ** (k / 40) for k in range(40, 40 * 5 + 1)]
    ax.plot(vols, [local_cost_per_correct(v, c_hw, life, e_kwh, p_el, loc.acc) for v in vols],
            color=LOCAL, label=f"Local, PC bought (${c_hw:,.0f}, {life} years)")
    owned = e_kwh * p_el / loc.acc
    ax.axhline(owned, color=LOCAL, ls=(0, (1, 1.5)), label="Local, PC already owned (electricity only)")
    ax.axhline(son.cost_usd_mean / son.acc, color=SONNET, ls="--", label="Claude Sonnet 5.5 API")
    ax.axhline(hai.cost_usd_mean / hai.acc, color=HAIKU, ls="-.", label="Claude Haiku 5.5 API")
    ax.axvline(v_max, color=MUTED, lw=1, ls=":")
    ax.text(v_max * 0.92, 2e-2, f"capacity limit\n{v_max:,.0f} tasks/day", ha="right", va="top", color=MUTED, fontsize=8)
    v_son = break_even_volume(c_hw, life, e_kwh, p_el, loc.acc, son.cost_usd_mean, son.acc)
    ax.plot([v_son], [son.cost_usd_mean / son.acc], "o", ms=7, color=SONNET, mec="white", mew=1.5, zorder=5)
    ax.annotate(f"break-even ≈ {v_son:,.0f}/day", (v_son, son.cost_usd_mean / son.acc), xytext=(8, 10),
                textcoords="offset points", color=INK, fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel="tasks per day", ylabel="USD per correctly solved task",
           xlim=(10, 1e5), ylim=(1e-5, 5e-2))
    ax.xaxis.set_major_formatter(count)
    ax.yaxis.set_major_formatter(money)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
    fig.savefig(OUT / "fig1_cost_vs_volume.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    prices = list(range(300, 3001, 25))
    lo_p, hi_p = min(eco.electricity_usd_per_kwh), max(eco.electricity_usd_per_kwh)
    for api, color, ls, name in ((son, SONNET, "--", "Sonnet 5.5"), (hai, HAIKU, "-.", "Haiku 5.5")):
        def curve(p: float) -> list[float]:
            return [break_even_volume(c, life, e_kwh, p, loc.acc, api.cost_usd_mean, api.acc) for c in prices]
        ax.fill_between(prices, curve(lo_p), curve(hi_p), color=color, alpha=0.15, lw=0)
        ax.plot(prices, curve(p_el), color=color, ls=ls, label=f"vs {name} (electricity ${p_el:.2f}/kWh)")
    ax.axhline(v_max, color=MUTED, lw=1, ls=":")
    ax.text(prices[0] + 20, v_max * 1.15, f"capacity limit {v_max:,.0f} tasks/day", color=MUTED, fontsize=8)
    ax.set(yscale="log", xlabel="hardware price, USD", ylabel="break-even volume, tasks/day", ylim=(50, 3e5))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.yaxis.set_major_formatter(count)
    ax.legend(loc="upper left", fontsize=8)
    ax.text(0.99, 0.02, f"bands: electricity \\${lo_p:.2f}–\\${hi_p:.2f}/kWh; lifetime {life} years",
            transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=7.5)
    fig.savefig(OUT / "fig2_breakeven_vs_price.png")
    plt.close(fig)

    tps = [float(r["gen_tokens_per_s"]) for r in rows if r["model_name"] == "qwen3-8b" and r["gen_tokens_per_s"]]
    import json
    weights = next(json.loads(l)["backend_info"]["weights_bytes"]
                   for l in (ROOT / "results" / "runs.jsonl").read_text(encoding="utf-8").splitlines()
                   if l and json.loads(l)["model"]["name"] == "qwen3-8b")
    ceiling = float(cfg.hardware["memory_bandwidth_gbs"]) * 1e9 / weights
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.hist(tps, bins=24, color=LOCAL, edgecolor="white", linewidth=0.8)
    ax.axvline(ceiling, color=INK, lw=1.5, ls="--")
    ax.text(ceiling - 0.4, ax.get_ylim()[1] * 0.95, f"bandwidth ceiling\n{ceiling:.1f} tokens/s",
            ha="right", va="top", fontsize=8, color=INK)
    mean = st.fmean(tps)
    ax.axvline(mean, color=MUTED, lw=1)
    ax.text(mean - 0.6, ax.get_ylim()[1] * 0.95, f"mean {mean:.1f}", ha="right", va="top", fontsize=8, color=MUTED)
    ax.set(xlabel="generation speed of qwen3:8b, tokens/s (one bar = tasks in that range)", ylabel="tasks",
           xlim=(min(tps) - 2, ceiling + 3))
    ax.grid(axis="x", visible=False)
    fig.savefig(OUT / "fig3_speed_vs_ceiling.png")
    plt.close(fig)

    print(f"figures -> {OUT}")
    print(f"E_task {e_kwh * 1000:.4f} Wh; owned local {owned:.6f} $/correct; V*(Sonnet) {v_son:.0f}; "
          f"V_max {v_max:.0f}; ceiling {ceiling:.1f}; mean tps {mean:.1f}")


if __name__ == "__main__":
    main()
