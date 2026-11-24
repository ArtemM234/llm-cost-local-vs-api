"""llmcost run | selftest | analyze | power-probe"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

from .checkers import check
from .config import load_config, load_prices
from .tasks import SOURCES, load_tasks, select


def _tasks(args: argparse.Namespace):
    return select(load_tasks(args.tasks, Path(args.data)), args.limit, args.offset)


async def _run(args: argparse.Namespace) -> None:
    from .runner import ResultWriter, run_api, run_local

    cfg = load_config(Path(args.config))
    prices = load_prices(Path(args.prices))
    tasks = _tasks(args)
    names = args.models.split(",") if args.models else [m.name for m in cfg.models]
    models = [cfg.model(n) for n in names]
    w = ResultWriter(Path(args.out))
    print(f"Набор {args.tasks}: {len(tasks)} задач × {args.repeat} повтор(ов); цены от {prices.fixed_on}")
    for m in models:
        if m.backend == "anthropic":
            await run_api(m, tasks, args.repeat, cfg, prices, w)
        else:
            await run_local(m, tasks, args.repeat, cfg, w, energy=not args.no_energy)


def _selftest(args: argparse.Namespace) -> int:
    """Checker sanity: reference answers must pass, obviously wrong ones must fail."""
    data = Path(args.data)
    failures = 0
    gsm = load_tasks("gsm8k", data)
    bad_gsm = [t.task_id for t in gsm if not check(t, f"...\nANSWER: {t.reference['answer']}").correct]
    wrong_gsm = [t.task_id for t in gsm[:50] if check(t, "ANSWER: -123456789").correct]
    print(f"GSM8K: эталон не принят {len(bad_gsm)}/{len(gsm)}, заведомо неверный принят {len(wrong_gsm)}/50")
    failures += len(bad_gsm) + len(wrong_gsm)

    he = load_tasks("humaneval_plus", data)

    def ref(t):
        r = t.reference
        return t.task_id, check(t, f"```python\n{r['prompt']}{r['canonical_solution']}\n```", 120)

    def stub(t):  # signature with `pass` must fail on every task
        r = t.reference
        return t.task_id, check(t, f"```python\ndef {r['entry_point']}(*args, **kwargs):\n    return None\n```", 120)

    t0 = time.time()
    with ThreadPoolExecutor(args.jobs) as ex:
        ref_res = list(ex.map(ref, he))
        stub_res = list(ex.map(stub, he))
    bad_ref = [(i, r.detail) for i, r in ref_res if not r.correct]
    bad_stub = [i for i, r in stub_res if r.correct]
    print(f"HumanEval+: эталон не принят {len(bad_ref)}/{len(he)}, заглушка принята {len(bad_stub)}/{len(he)} "
          f"({time.time() - t0:.0f} с)")
    for i, d in bad_ref:
        print(f"  {i}: {d}")
    failures += len(bad_ref) + len(bad_stub)
    print("OK" if not failures else f"ПРОВАЛ: {failures}")
    return 1 if failures else 0


async def _power_probe(seconds: float) -> None:
    from .energy import make_sampler

    s = make_sampler()
    print(f"датчик: {s.source}")
    await s.start()
    await asyncio.sleep(seconds)
    await s.stop()
    for p in s.samples[-10:]:
        print(f"  {p.duration_s * 1000:.0f} ms  {p.watts:.2f} W")
    print(f"образцов: {len(s.samples)}, не распознано: {s.parse_failures}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="llmcost")
    p.add_argument("--data", default="data")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="прогнать задачи через модели")
    r.add_argument("--tasks", choices=list(SOURCES), required=True)
    r.add_argument("--models", help="имена через запятую из config.toml (по умолчанию все)")
    r.add_argument("--limit", type=int)
    r.add_argument("--offset", type=int, default=0)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--no-energy", action="store_true", help="без powermetrics (не macOS или отладка)")
    r.add_argument("--config", default="config.toml")
    r.add_argument("--prices", default="prices.toml")
    r.add_argument("--out", default="results")

    s = sub.add_parser("selftest", help="проверить проверяльщики на эталонных решениях")
    s.add_argument("--jobs", type=int, default=8)

    a = sub.add_parser("analyze", help="сводка, V*, проверка bandwidth")
    a.add_argument("--config", default="config.toml")
    a.add_argument("--out", default="results")

    pp = sub.add_parser("power-probe", help="проверить чтение powermetrics (macOS, нужен sudo -v)")
    pp.add_argument("--seconds", type=float, default=5)

    args = p.parse_args(argv)
    match args.cmd:
        case "run":
            asyncio.run(_run(args))
        case "selftest":
            return _selftest(args)
        case "analyze":
            from .analyze import analyze

            analyze(Path(args.out), load_config(Path(args.config)))
        case "power-probe":
            asyncio.run(_power_probe(args.seconds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
