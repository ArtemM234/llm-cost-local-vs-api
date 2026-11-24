"""End-to-end with fake backends: run → CSV → analyze, no network and no Mac needed."""

import asyncio
import csv
from pathlib import Path

import pytest

from llmcost import runner
from llmcost.analyze import analyze
from llmcost.backends import GenResult
from llmcost.config import load_config, load_prices
from llmcost.tasks import Task

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self, cfg, run, right: bool):
        self.cfg, self.right = cfg, right

    async def generate(self, prompt: str) -> GenResult:
        if "single word OK" in prompt:
            return GenResult("OK", 5, 1, 0.01, "stop")
        answer = prompt.split("=")[-1] if self.right else "0"
        return GenResult(f"ANSWER: {answer}", 100, 50, 2.0, "end_turn", gen_tokens_per_s=40.0)

    async def info(self):
        return {"weights_bytes": 5_000_000_000}

    async def aclose(self):
        pass


@pytest.fixture
def tasks():
    return [Task("gsm8k", f"gsm8k/{i}", f"q {i} ={i + 1}", {"answer": str(i + 1)}) for i in range(4)]


def test_run_and_analyze(tmp_path, monkeypatch, tasks, capsys):
    cfg = load_config(ROOT / "config.toml")
    prices = load_prices(ROOT / "prices.toml")
    monkeypatch.setattr(runner, "make_backend",
                        lambda m, run: FakeBackend(m, run, right=m.backend == "anthropic" or m.name == "qwen3-8b"))
    w = runner.ResultWriter(tmp_path)

    async def go():
        await runner.run_api(cfg.model("haiku-5.5"), tasks, 1, cfg, prices, w)
        await runner.run_local(cfg.model("qwen3-8b"), tasks, 2, cfg, w, energy=False)
        await runner.run_api(cfg.model("haiku-5.5"), tasks, 1, cfg, prices, w)  # resume: nothing to do

    asyncio.run(go())
    rows = list(csv.DictReader((tmp_path / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 4 + 8
    api_row = next(r for r in rows if r["model_name"] == "haiku-5.5")
    assert float(api_row["cost_usd"]) == pytest.approx((100 * 0.10 + 50 * 0.50) / 1e6)
    assert float(api_row["cost_usd_batch"]) == pytest.approx(float(api_row["cost_usd"]) / 2)

    # wall meter: 0.4 Wh net over the run → 0.05 Wh per task (8 tasks)
    run_id = next(r["run_id"] for r in rows if r["model_name"] == "qwen3-8b")
    (tmp_path / "wall_meter.csv").write_text(f"run_id,wh_measured,idle_w\n{run_id},0.4,0\n", encoding="utf-8")
    analyze(tmp_path, cfg)
    grid = list(csv.DictReader((tmp_path / "breakeven.csv").open(encoding="utf-8")))
    assert grid and all(r["energy_source"] == "wall" for r in grid)
    assert "V*" in capsys.readouterr().out
