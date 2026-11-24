from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

TaskSetName = Literal["gsm8k", "humaneval_plus"]


@dataclass(frozen=True, slots=True)
class Task:
    task_set: TaskSetName
    task_id: str
    prompt: str
    reference: dict[str, Any] = field(repr=False)


@dataclass(frozen=True, slots=True)
class _Source:
    url: str
    filename: str
    sha256: str


SOURCES: dict[TaskSetName, _Source] = {
    "gsm8k": _Source(
        "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl",
        "gsm8k_test.jsonl",
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14",
    ),
    "humaneval_plus": _Source(
        "https://github.com/evalplus/humanevalplus_release/releases/download/v0.1.10/HumanEvalPlus.jsonl.gz",
        "HumanEvalPlus-v0.1.10.jsonl.gz",
        "272720b90ac375502c8ed23cd791c2a93dfb22a911641a494da74a426c09f101",
    ),
}

GSM8K_TEMPLATE = (
    "Solve the following math problem. You may reason step by step. "
    "On the last line write only the final numeric answer in the form:\n"
    "ANSWER: <number>\n\n"
    "Problem: {question}"
)

HUMANEVAL_TEMPLATE = (
    "Complete the following Python function. Return the complete function "
    "(signature, body and any imports it needs) in a single ```python code block. "
    "Do not include tests or example usage.\n\n"
    "```python\n{prompt}```"
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_file(name: TaskSetName, data_dir: Path) -> Path:
    src = SOURCES[name]
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / src.filename
    if not path.exists():
        tmp = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(src.url, tmp)
        tmp.replace(path)
    if (digest := _sha256(path)) != src.sha256:
        raise RuntimeError(
            f"{path}: SHA-256 {digest} != pinned {src.sha256}. Файл изменился у источника — "
            "удалите его и проверьте версию набора задач."
        )
    return path


def _gsm8k_answer(solution: str) -> str:
    return solution.rsplit("####", 1)[1].strip().replace(",", "")


def load_tasks(name: TaskSetName, data_dir: Path) -> list[Task]:
    path = ensure_file(name, data_dir)
    match name:
        case "gsm8k":
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
            return [
                Task(
                    task_set=name,
                    task_id=f"gsm8k/{i}",
                    prompt=GSM8K_TEMPLATE.format(question=r["question"]),
                    reference={"answer": _gsm8k_answer(r["answer"])},
                )
                for i, r in enumerate(rows)
            ]
        case "humaneval_plus":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            return [
                Task(
                    task_set=name,
                    task_id=r["task_id"],
                    prompt=HUMANEVAL_TEMPLATE.format(prompt=r["prompt"]),
                    reference={
                        "prompt": r["prompt"],
                        "entry_point": r["entry_point"],
                        "canonical_solution": r["canonical_solution"],
                        "inputs": r["base_input"] + r["plus_input"],
                        "atol": r["atol"],
                    },
                )
                for r in rows
            ]
    raise ValueError(f"unknown task set: {name}")


def select(tasks: list[Task], limit: int | None, offset: int = 0) -> list[Task]:
    end = None if limit is None else offset + limit
    return tasks[offset:end]
