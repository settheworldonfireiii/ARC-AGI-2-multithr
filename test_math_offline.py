"""Offline test for the math (AIME) domain — NO network, NO API keys.

Uses the bundled 'aime-sample' tasks and a mocked provider that always answers
\\boxed{16}, so the sample whose answer is 16 gets solved and the rest exercise
the retry path. Also checks the gold answer is never serialised into a prompt.
"""
import render  # noqa: F401
import orch_2
from orch_2 import MathOrchestrator
from math_dataset import MathTask, extract_final_answer, grade_answer

# --- anti-leak: a MathTask folded into a prompt must NOT expose the answer ---
leak = MathOrchestrator._as_text(MathTask(id="z", problem="What is X?", answer="999"))
assert "999" not in leak and "answer" not in leak, leak
print("anti-leak _as_text OK ->", leak)

# --- grading sanity ---
assert grade_answer(extract_final_answer("so \\boxed{16}"), "16")
assert not grade_answer(extract_final_answer("so \\boxed{15}"), "16")


# --- mocked provider: always returns a solution ending in \boxed{16} ---
def fake_raw_call(self, client, model, provider, content, max_tokens):
    return "Let me reason about this... therefore the answer is \\boxed{16}.", 300, 40


orch_2.Orchestrator._raw_call = fake_raw_call

orch = MathOrchestrator(
    "claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001", "hemens",
    "sk-ant-dummy", "sk-ant-dummy", "hemens",
    num_tasks=6, num_retries=1, max_workers=3, budget=None,
)
orch.dataset_source = "aime-sample"   # offline bundled tasks
res = orch.run()
print("math run result:", res)
assert res["num_tasks"] == 6
assert res["solved"] >= 1, "the \\boxed{16} answer should solve the divisors-of-2024 task"

print("ALL MATH OFFLINE TESTS PASSED")
