"""Offline smoke test for orch_2.py — NO network, NO API keys.

Monkeypatches Orchestrator._raw_call so we can exercise the multithreaded
scheduler, the BudgetTracker, the curriculum retry loop and the context-append
fixes without spending tokens.

Run:  .venv/bin/python test_offline.py
"""
import threading

# Importing render first forces orch_2 to fully initialise through the
# (pre-existing) circular import render <-> orch_2.
import render  # noqa: F401
import orch_2
from orch_2 import Orchestrator
import pricing

# ---- fake provider call: returns trivial code + fixed token usage ----
_threads_seen = set()
_call_count = {"n": 0}
_count_lock = threading.Lock()


def fake_raw_call(self, client, model, provider, content, max_tokens):
    _threads_seen.add(threading.current_thread().name)
    with _count_lock:
        _call_count["n"] += 1
    text = "```python\ndef transform(grid):\n    return grid\n```"
    return text, 1200, 180  # (text, input_tokens, output_tokens)


# Force a deterministic mix of solved/unsolved to exercise every branch:
# even task index "solves", odd index "fails" -> retry/cluster/compress paths.
def fake_parse_output(self, output, task):
    # Deterministic ~half split by task id (hex stem) so we exercise BOTH the
    # solved path (succeeded/select_cluster/compress) and the retry path.
    try:
        return 1.0 if (int(task.id, 16) % 2 == 0) else 0.0
    except Exception:
        return 0.0


Orchestrator._raw_call = fake_raw_call
Orchestrator.parse_output = fake_parse_output


def main():
    print("=== TEST 1: multithreaded, no budget ===")
    orch = Orchestrator(
        "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", "hemens",
        "sk-ant-dummy", "sk-ant-dummy", "hemens",
        num_tasks=6, num_retries=2, max_workers=4, max_output_tokens=4096,
        budget=None,
    )
    res = orch.run()
    print("result:", res)
    print("distinct worker threads used:", len(_threads_seen), sorted(_threads_seen))
    assert len(_threads_seen) > 1, "expected real multithreading (>1 thread)"
    assert _call_count["n"] > 0

    print("\n=== TEST 2: budget enforced (tiny cap, should stop early) ===")
    _threads_seen.clear()
    budget = pricing.BudgetTracker(max_usd=0.01)  # ~ stops after a few calls
    orch2 = Orchestrator(
        "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", "hemens",
        "sk-ant-dummy", "sk-ant-dummy", "hemens",
        num_tasks=8, num_retries=3, max_workers=4, max_output_tokens=2048,
        budget=budget,
    )
    res2 = orch2.run()
    print("result:", res2)
    print(budget.summary())
    assert budget.spent_usd > 0, "budget should have recorded spend"
    assert budget.would_exceed(), "budget should be marked exhausted"

    print("\n=== TEST 3: _as_text coerces Challenge / list (the old crash) ===")
    ch = orch_2.Challenge(id="x", train=[], test=[])
    assert isinstance(Orchestrator._as_text(ch), str)
    assert Orchestrator._as_text([{"a": 1}]) == '[{"a": 1}]'
    assert Orchestrator._as_text("hi") == "hi"
    assert Orchestrator._as_text(None) == ""
    print("_as_text OK")

    print("\nALL OFFLINE TESTS PASSED")


if __name__ == "__main__":
    main()
