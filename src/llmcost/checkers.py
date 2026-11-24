from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .tasks import Task

_CHILD = Path(__file__).with_name("_sandbox_child.py")


@dataclass(frozen=True, slots=True)
class CheckResult:
    correct: bool
    extracted: str
    detail: str = ""


_ANSWER_RE = re.compile(r"ANSWER\s*[:：]\s*[*$\\{(\s]*(-?[\d,]*\.?\d+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _to_number(s: str) -> float | None:
    try:
        return float(s.replace(",", "").rstrip("."))
    except ValueError:
        return None


def extract_number(text: str) -> str | None:
    if found := _ANSWER_RE.findall(text):
        return found[-1].replace(",", "")
    if found := _NUMBER_RE.findall(text):
        return found[-1].replace(",", "")
    return None


def check_gsm8k(task: Task, output: str) -> CheckResult:
    got = extract_number(output)
    if got is None:
        return CheckResult(False, "", "no number in output")
    a, b = _to_number(got), _to_number(task.reference["answer"])
    ok = a is not None and b is not None and abs(a - b) < 1e-6
    return CheckResult(ok, got)


_FENCE_RE = re.compile(r"```[ \t]*(?:python|py|python3)?[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(output: str, entry_point: str) -> str:
    blocks = _FENCE_RE.findall(output)
    if not blocks:
        start = output.find("```")
        blocks = [output[output.index("\n", start) + 1 :]] if start != -1 and "\n" in output[start:] else [output]
    with_def = [b for b in blocks if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", b)]
    return (with_def or blocks)[-1]


def build_candidate(prompt: str, code: str, entry_point: str) -> str:
    if re.search(rf"def\s+{re.escape(entry_point)}\s*\(", code):
        return prompt + "\n\n" + code
    return prompt + code


def run_code_check(task_id: str, ref: dict, candidate: str, timeout_s: float) -> dict:
    spec = {
        "task_id": task_id,
        "entry_point": ref["entry_point"],
        "canonical": ref["prompt"] + ref["canonical_solution"],
        "candidate": candidate,
        "inputs": ref["inputs"],
        "atol": ref["atol"],
        "cpu_limit_s": int(timeout_s),
    }
    with tempfile.TemporaryDirectory(prefix="llmcost_") as tmp:
        spec_path, result_path = Path(tmp, "spec.json"), Path(tmp, "result.json")
        spec_path.write_text(json.dumps(spec), encoding="utf-8")
        try:
            subprocess.run(
                [sys.executable, "-I", str(_CHILD), str(spec_path), str(result_path)],
                cwd=tmp,
                timeout=timeout_s,
                capture_output=True,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"timeout {timeout_s}s"}
        if not result_path.exists():
            return {"passed": False, "error": "child died without result (exit/crash/CPU limit)"}
        return json.loads(result_path.read_text(encoding="utf-8"))


def check_humaneval(task: Task, output: str, timeout_s: float = 60.0) -> CheckResult:
    ref = task.reference
    code = extract_code(output, ref["entry_point"])
    candidate = build_candidate(ref["prompt"], code, ref["entry_point"])
    res = run_code_check(task.task_id, ref, candidate, timeout_s)
    return CheckResult(bool(res["passed"]), code, res.get("error", ""))


def check(task: Task, output: str, code_timeout_s: float = 60.0) -> CheckResult:
    match task.task_set:
        case "gsm8k":
            return check_gsm8k(task, output)
        case "humaneval_plus":
            return check_humaneval(task, output, code_timeout_s)
    raise ValueError(task.task_set)
