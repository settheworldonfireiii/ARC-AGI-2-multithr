"""Validate the orch_2.py CLI budget wiring end-to-end (no network).

run() is stubbed so we only check that plan -> Orchestrator construction
applies the right num_retries / max_output_tokens / budget.
Run non-interactively (stdin = /dev/null) so the decline branch is hit via EOF.
"""
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-dummy"  # force (env may have it empty)

import render  # noqa: F401
import orch_2

captured = {}


def fake_run(self):
    captured.clear()
    captured.update(
        num_retries=self.num_retries,
        max_output_tokens=self.max_output_tokens,
        budget_cap=(None if self.budget is None else self.budget.max_usd),
        workers=self.max_workers,
        num_tasks=self.num_tasks,
    )
    return {"solved": 0, "num_tasks": self.num_tasks}


orch_2.Orchestrator.run = fake_run

print("--- case A: budget only -> derive trials, budget ON ---")
orch_2.main(["--budget-usd", "5", "--num-tasks", "40", "--workers", "4"])
print(captured)
assert captured["budget_cap"] == 5 and captured["workers"] == 4
assert captured["num_retries"] >= 1

print("--- case B: inconsistent (both set), --yes -> balanced trials ---")
orch_2.main(["--budget-usd", "1", "--num-trials", "50",
             "--context-length", "8192", "--num-tasks", "40", "--yes"])
print(captured)
assert captured["num_retries"] == 1, captured        # 50 -> 1 balanced
assert captured["budget_cap"] == 1
assert captured["max_output_tokens"] == 8192

print("--- case C: inconsistent, declined -> budget OFF, keep user values ---")
# Deterministic decline regardless of TTY: inject prompter -> 'n'.
# (Non-interactive runs also decline via the isatty guard in resolve_budget.)
import functools
_orig_rb = orch_2.resolve_budget
orch_2.resolve_budget = functools.partial(_orig_rb, prompter=lambda *_: "n")
try:
    orch_2.main(["--budget-usd", "1", "--num-trials", "50",
                 "--context-length", "8192", "--num-tasks", "40"])
finally:
    orch_2.resolve_budget = _orig_rb
print(captured)
assert captured["num_retries"] == 50, captured       # user value kept
assert captured["budget_cap"] is None                # budget disabled
assert captured["max_output_tokens"] == 8192

print("CLI OFFLINE TESTS PASSED")
