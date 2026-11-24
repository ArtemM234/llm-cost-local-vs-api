# Local language model or Claude API: cost of a correct answer

Code and data for the paper *Local Language Model or Claude API: The Cost of a Correct Answer on a Home Computer*
(Artem Matyrin, Vistula University, Architecture of Computers, Winter 2026/2027).

The harness gives the first 200 GSM8K test problems to a local model (qwen3:8b via Ollama) and to Claude Haiku 5.5
and Claude Sonnet 5.5 via the Anthropic API, with one prompt and one automatic checker. It records correctness,
tokens, time, API cost and graphics card power (nvidia-smi, minus idle), then computes the cost per correct
answer and the break-even daily volume.

## Contents

| Path | What it is |
|---|---|
| `src/llmcost/` | harness: tasks and checker, backends, energy sampling, runner, cost model, analysis |
| `tests/` | unit tests (`pytest`) |
| `config.toml` | models, run settings, hardware and economic assumptions |
| `prices.toml` | API prices per million tokens, checked on 2026-11-23 |
| `scripts/run_windows.ps1` | full run on Windows: API models, local model, analysis |
| `results/results.csv` | one row per task and model (correct, tokens, time, energy, cost) |
| `results/raw.jsonl` | full model answers |
| `results/runs.jsonl` | run metadata (host, versions, idle power) |
| `results/summary.csv`, `results/breakeven.csv` | per-model summary and break-even grid used in the paper |
| `paper/make_figures.py`, `paper/build_paper.py` | figures and paper, computed from the result files |

## Reproduce

Requirements: Python 3.12+, an NVIDIA GPU with nvidia-smi, Ollama with `qwen3:8b`, an Anthropic API key.

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]" matplotlib==3.11.2 python-docx==1.2.0
copy .env.example .env        # then put your ANTHROPIC_API_KEY into .env
ollama pull qwen3:8b
.venv\Scripts\python -m pytest -q
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 -Limit 200
.venv\Scripts\llmcost analyze
.venv\Scripts\python paper\make_figures.py
```

GSM8K is downloaded on first use from the original repository and checked against a fixed SHA-256 hash.
A run resumes where it stopped: finished task and model pairs in `results/results.csv` are skipped.

## Notes

- Claude 5.5 models do not accept a non-default temperature, so only the local model runs at temperature 0.
- Energy covers the graphics card only, not the whole computer.
- Two GSM8K references (tasks 119 and 184) are doubtful; scoring was left unchanged (see the paper, Section 4.4).
