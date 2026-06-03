"""
math_dataset.py — math-competition datasets (AIME / olympiad style) for the
orchestrator's `--dataset` switch, plus answer extraction and grading.

Why AIME rather than IMO proper: IMO problems are proof-based and cannot be
auto-graded by exact match. AIME (the AMC→AIME→USAMO→IMO competition lineage)
is popular, has a famously low model solve rate, and every answer is a single
integer 0–999 — so it grades exactly, which matches this orchestrator's
"score == fraction correct" design.

Loading (no heavy `datasets` dependency required):
  * a local .json / .jsonl file of records  -> use it;
  * a HuggingFace dataset id "org/name"      -> fetched via the public
    datasets-server API (urllib only);
  * "aime-sample" / "sample"                  -> a small bundled set so things
    run offline.

Records are normalised to MathTask(id, problem, answer, solution?).
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request

from pydantic import BaseModel


class MathTask(BaseModel):
    id: str
    problem: str
    answer: str
    solution: str | None = None


# Popular, low-solve-rate default. Override with --dataset-path or another id.
DEFAULT_AIME_DATASET = "Maxwell-Jia/AIME_2024"

# Column-name candidates (datasets disagree on capitalisation / naming).
_PROBLEM_KEYS = ["problem", "Problem", "question", "Question", "problem_statement"]
_ANSWER_KEYS = ["answer", "Answer", "final_answer", "solution_answer", "gt_answer"]
_ID_KEYS = ["id", "ID", "problem_id", "index", "uid"]
_SOLUTION_KEYS = ["solution", "Solution", "rationale", "explanation"]


# --- a tiny bundled set (correct integer answers) for OFFLINE use/testing. ---
# NOTE: these are illustrative number-theory/algebra items, not official AIME
# problems. Use --dataset aime (HF) or --dataset-path for the real benchmark.
_SAMPLE: list[MathTask] = [
    MathTask(id="s1", problem="How many positive divisors does 2024 have?", answer="16"),
    MathTask(id="s2", problem="What is the remainder when 7^100 is divided by 5?", answer="1"),
    MathTask(id="s3", problem="How many positive integers less than 100 are divisible by both 3 and 4?", answer="8"),
    MathTask(id="s4", problem="Compute the sum of the first 20 positive integers.", answer="210"),
    MathTask(id="s5", problem="What is the units digit of 3^2024?", answer="1"),
    MathTask(id="s6", problem="How many integers from 1 to 1000 inclusive are perfect squares?", answer="31"),
]


def _first_key(row: dict, keys: list[str]):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _row_to_task(row: dict, fallback_id: str) -> MathTask | None:
    problem = _first_key(row, _PROBLEM_KEYS)
    if problem is None:
        return None
    answer = _first_key(row, _ANSWER_KEYS)
    solution = _first_key(row, _SOLUTION_KEYS)
    if answer is None and solution is not None:
        answer = extract_final_answer(str(solution))
    if answer is None:
        return None
    tid = _first_key(row, _ID_KEYS)
    return MathTask(
        id=str(tid) if tid is not None else fallback_id,
        problem=str(problem),
        answer=str(answer).strip(),
        solution=str(solution) if solution is not None else None,
    )


def _load_local(path: str, num_tasks: int) -> list[MathTask]:
    out: list[MathTask] = []
    with open(path) as f:
        if path.endswith(".jsonl"):
            rows = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            rows = data if isinstance(data, list) else data.get("rows", data.get("data", []))
    for i, row in enumerate(rows):
        if isinstance(row, dict) and "row" in row and isinstance(row["row"], dict):
            row = row["row"]  # datasets-server style
        t = _row_to_task(row, fallback_id=f"task{i}")
        if t is not None:
            out.append(t)
        if len(out) >= num_tasks:
            break
    return out


def _http_json(url: str, timeout: float = 20.0):
    req = urllib.request.Request(url, headers={"User-Agent": "orch2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _load_hf_via_api(dataset: str, num_tasks: int,
                     config: str | None, split: str | None) -> list[MathTask]:
    """Fetch rows from the HuggingFace datasets-server (no `datasets` lib)."""
    base = "https://datasets-server.huggingface.co"
    if config is None or split is None:
        info = _http_json(f"{base}/splits?dataset={urllib.parse.quote(dataset)}")
        splits = info.get("splits", [])
        if not splits:
            raise RuntimeError(f"no splits found for HF dataset '{dataset}'")
        # prefer a test/train split if not specified
        chosen = splits[0]
        for s in splits:
            if split and s.get("split") == split:
                chosen = s
                break
        config = config or chosen["config"]
        split = split or chosen["split"]
    rows: list[MathTask] = []
    offset = 0
    while len(rows) < num_tasks:
        length = min(100, num_tasks - len(rows))
        url = (f"{base}/rows?dataset={urllib.parse.quote(dataset)}"
               f"&config={urllib.parse.quote(config)}"
               f"&split={urllib.parse.quote(split)}"
               f"&offset={offset}&length={length}")
        data = _http_json(url)
        batch = data.get("rows", [])
        if not batch:
            break
        for r in batch:
            row = r.get("row", r)
            t = _row_to_task(row, fallback_id=f"{dataset.replace('/', '_')}_{offset}")
            if t is not None:
                rows.append(t)
        offset += len(batch)
        if len(batch) < length:
            break
    return rows[:num_tasks]


def load_math_tasks(source: str, num_tasks: int,
                    config: str | None = None, split: str | None = None) -> list[MathTask]:
    """Load up to `num_tasks` MathTasks from a local file, a HF id, or the sample."""
    if source in ("sample", "aime-sample", None):
        return _SAMPLE[:num_tasks]
    if os.path.exists(source):
        return _load_local(source, num_tasks)
    if source in ("aime", "AIME"):
        source = DEFAULT_AIME_DATASET
    # treat as a HuggingFace dataset id
    try:
        tasks = _load_hf_via_api(source, num_tasks, config, split)
        if not tasks:
            raise RuntimeError("0 rows parsed")
        return tasks
    except Exception as e:
        raise RuntimeError(
            f"Could not load math dataset '{source}' ({e}). "
            f"Use --dataset-path <file.jsonl>, a different HF id, or "
            f"--dataset aime-sample for the offline sample."
        ) from e


# ---------------------------------------------------------------------------
# Answer extraction + grading
# ---------------------------------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    """Return the contents of the LAST \\boxed{...} with balanced braces."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_final_answer(text: str) -> str:
    """Pull a final answer out of free-form model output."""
    if not text:
        return ""
    boxed = _extract_boxed(text)
    if boxed is not None:
        return boxed.strip()
    m = re.search(r"(?:final answer|the answer)\s*(?:is|:|=)?\s*\$?\s*([+-]?\d+(?:\.\d+)?)",
                  text, re.IGNORECASE)
    if m:
        return m.group(1)
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else text.strip()


def _normalize(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\boxed", "").replace("\\!", "").replace("\\,", "")
    s = s.replace("$", "").replace("\\(", "").replace("\\)", "")
    s = s.strip().strip("{}").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace(",", "").replace(" ", "")
    if s.endswith("."):
        s = s[:-1]
    return s


def _equal(p: str, g: str) -> bool:
    if p == g and p != "":
        return True
    try:
        if int(p) == int(g):
            return True
    except Exception:
        pass
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except Exception:
        pass
    try:  # last resort: symbolic equality (sympy is optional)
        import sympy
        from sympy.parsing.sympy_parser import parse_expr
        if sympy.simplify(parse_expr(p) - parse_expr(g)) == 0:
            return True
    except Exception:
        pass
    return False


def grade_answer(pred: str, gold: str) -> bool:
    """True if `pred` matches `gold` (string, int, float, or sympy-equal).

    Tries the raw prediction first, then an answer extracted from it, so it
    works whether it receives a clean answer or a full solution string.
    """
    g = _normalize(gold)
    candidates = [pred]
    extracted = extract_final_answer(str(pred))
    if extracted and extracted != pred:
        candidates.append(extracted)
    return any(_equal(_normalize(c), g) for c in candidates)


if __name__ == "__main__":
    # offline self-test (no network)
    assert extract_final_answer("blah \\boxed{42} done") == "42"
    assert extract_final_answer("so the final answer is 17.") == "17"
    assert extract_final_answer("...therefore 0090") == "0090"
    assert grade_answer("\\boxed{16}", "16")
    assert grade_answer("016", "16")
    assert grade_answer("the answer is 8", "8")
    assert not grade_answer("\\boxed{15}", "16")
    print("math_dataset self-test OK;", len(_SAMPLE), "sample tasks")
