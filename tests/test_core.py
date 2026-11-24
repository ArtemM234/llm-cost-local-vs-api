import math
from pathlib import Path

import pytest

from llmcost.checkers import build_candidate, check_gsm8k, extract_code, extract_number
from llmcost.costmodel import break_even_volume, local_cost_per_correct, max_daily_volume, wilson_interval
from llmcost.energy import PowerSample, integrate, parse_sample
from llmcost.tasks import Task


def gsm(answer: str) -> Task:
    return Task("gsm8k", "gsm8k/x", "", {"answer": answer})


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so 9*2=18\nANSWER: 18", "18"),
        ("ANSWER: $1,250.00", "1250.00"),
        ("ANSWER: **42**", "42"),
        ("answer: -7", "-7"),
        ("first ANSWER: 3 then corrected ANSWER: 5", "5"),
        ("The result is 12 apples, total 30.", "30"),
        ("no digits here", None),
    ],
)
def test_extract_number(text, expected):
    assert extract_number(text) == expected


def test_gsm8k_numeric_equality():
    assert check_gsm8k(gsm("1250"), "ANSWER: 1,250.00").correct
    assert not check_gsm8k(gsm("18"), "ANSWER: 17").correct


def test_extract_code_prefers_block_with_entry_point():
    out = "```python\nimport math\n```\ntext\n```python\ndef f(x):\n    return x\n```\n```python\nprint(f(1))\n```"
    assert extract_code(out, "f").startswith("def f(")


def test_extract_code_unterminated_fence():
    assert extract_code("```python\ndef f(x):\n    return x\n", "f").startswith("def f(")


def test_build_candidate_body_only_is_completion():
    prompt = "def f(x):\n    \"\"\"doc\"\"\"\n"
    assert build_candidate(prompt, "    return x\n", "f") == prompt + "    return x\n"


def test_break_even_matches_direct_cost_comparison():
    args = dict(c_hw=2000, lifetime_years=4, e_task_kwh=0.0005, p_el=0.2, acc_local=0.7)
    c_api, acc_api = 0.002, 0.95
    v = break_even_volume(**args, c_task_api=c_api, acc_api=acc_api)
    assert math.isfinite(v)
    assert local_cost_per_correct(v, **args) == pytest.approx(c_api / acc_api)


def test_break_even_never_when_energy_alone_exceeds_api():
    assert break_even_volume(2000, 4, 0.01, 0.3, 0.5, 0.0001, 0.9) == math.inf


def test_vmax_and_wilson():
    assert max_daily_volume(86.4) == pytest.approx(1000)
    lo, hi = wilson_interval(80, 100)
    assert lo == pytest.approx(0.711, abs=1e-3) and hi == pytest.approx(0.867, abs=1e-3)


def test_parse_sample_variants():
    assert parse_sample({"elapsed_ns": 200e6, "processor": {"combined_power": 5500}}, 0.2) == (0.2, 5.5)
    assert parse_sample({"processor": {"cpu_power": 1000, "gpu_power": 3000, "ane_power": 0}}, 0.2) == (0.2, 4.0)
    assert parse_sample({"elapsed_ns": 500e6, "processor": {"cpu_energy": 1000}}, 0.2) == (0.5, 2.0)
    assert parse_sample({"processor": {}}, 0.2) is None


def test_parse_nvidia_line():
    from llmcost.energy import parse_nvidia_line

    assert parse_nvidia_line("123.45\r\n") == 123.45
    assert parse_nvidia_line("[N/A]") is None


def test_integrate_partial_overlap():
    samples = [PowerSample(1.0, 1.0, 10.0), PowerSample(2.0, 1.0, 20.0)]
    joules, covered = integrate(samples, 0.5, 1.5)
    assert joules == pytest.approx(10 * 0.5 + 20 * 0.5)
    assert covered == pytest.approx(1.0)


DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.skipif(not (DATA / "HumanEvalPlus-v0.1.10.jsonl.gz").exists(), reason="dataset not downloaded")
def test_humaneval_wrong_solution_fails_and_reference_passes():
    from llmcost.checkers import check
    from llmcost.tasks import load_tasks

    t = load_tasks("humaneval_plus", DATA)[0]
    wrong = "```python\ndef has_close_elements(numbers, threshold):\n    return False\n```"
    assert not check(t, wrong, 30).correct
    right = ("```python\ndef has_close_elements(numbers, threshold):\n"
             "    s = sorted(numbers)\n    return any(b - a < threshold for a, b in zip(s, s[1:]))\n```")
    assert check(t, right, 30).correct
