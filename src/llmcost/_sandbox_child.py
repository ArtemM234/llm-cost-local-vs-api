import copy
import io
import json
import math
import sys


def _is_floats(x: object) -> bool:
    if isinstance(x, float):
        return True
    return isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(i, float) for i in x)


def _allclose(got: object, exp: object, atol: float, rtol: float = 1e-7) -> bool:
    def close(a: object, b: object) -> bool:
        if isinstance(a, bool) or not isinstance(a, (int, float)):
            return False
        return abs(a - b) <= atol + rtol * abs(b)

    if isinstance(exp, float):
        return close(got, exp)
    if not isinstance(got, (list, tuple)) or len(got) != len(exp):
        return False
    return all(close(a, b) for a, b in zip(got, exp))


def _poly(xs: list, x: float) -> float:
    return math.fsum(c * math.pow(x, i) for i, c in enumerate(xs))


def outputs_match(task_id: str, inp: list, got: object, exp: object, atol: float) -> bool:
    if task_id == "HumanEval/32":
        if isinstance(got, float) and isinstance(exp, float) and abs(got - exp) <= atol:
            return True
        try:
            return abs(_poly(*inp, got)) <= atol
        except Exception:
            return False
    if atol == 0 and _is_floats(exp):
        atol = 1e-6
    if got == exp:
        return True
    if atol != 0 and _is_floats(exp):
        return _allclose(got, exp, atol)
    return False


def run(spec: dict) -> dict:
    entry = spec["entry_point"]
    inputs = spec["inputs"]
    atol = float(spec["atol"])

    ref_ns: dict = {}
    exec(compile(spec["canonical"], "<canonical>", "exec"), ref_ns)
    ref_fn = ref_ns[entry]
    expected = []
    for inp in inputs:
        try:
            expected.append(("ok", ref_fn(*copy.deepcopy(inp))))
        except BaseException as e:
            expected.append(("exc", type(e).__name__))

    real_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        ns: dict = {}
        try:
            exec(compile(spec["candidate"], "<candidate>", "exec"), ns)
            fn = ns[entry]
        except BaseException as e:
            return {"passed": False, "n_passed": 0, "n_total": len(inputs), "error": f"load: {type(e).__name__}: {e}"[:300]}

        for i, (inp, (kind, exp)) in enumerate(zip(inputs, expected)):
            sys.stdout.seek(0)
            sys.stdout.truncate()
            try:
                got = fn(*copy.deepcopy(inp))
            except BaseException as e:
                if kind == "exc":
                    continue
                return {"passed": False, "n_passed": i, "n_total": len(inputs), "error": f"input {i}: {type(e).__name__}: {e}"[:300]}
            if kind == "exc" or not outputs_match(spec["task_id"], inp, got, exp, atol):
                return {"passed": False, "n_passed": i, "n_total": len(inputs), "error": f"input {i}: wrong output"}
        return {"passed": True, "n_passed": len(inputs), "n_total": len(inputs), "error": ""}
    finally:
        sys.stdout = real_stdout


def main() -> None:
    spec_path, result_path = sys.argv[1], sys.argv[2]
    with open(spec_path, encoding="utf-8") as f:
        spec = json.load(f)
    try:
        import resource

        cpu = int(spec.get("cpu_limit_s", 60))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
    except (ImportError, ValueError, OSError):
        pass
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10_000))
    result = run(spec)
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f)


if __name__ == "__main__":
    main()
