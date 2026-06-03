"""Offline test for the cloud orchestrator — NO server, NO network."""
import time

import render  # noqa: F401  (forces orch_2 to load)
import orch_2  # noqa: F401
from orch_2_cloud import CloudOrchestrator
from cloud_inference import EndpointBackend, WallClockBudget


# ---- _to_text flattening ----
assert CloudOrchestrator._to_text("hi") == "hi"
assert CloudOrchestrator._to_text(
    [{"type": "text", "text": "a"},
     {"type": "image_url", "image_url": {}},
     {"type": "text", "text": "b"}]
) == "a\nb"
print("_to_text OK")


# ---- fake OpenAI-compatible client (Chat Completions shape) ----
class _Usage:
    prompt_tokens = 50
    completion_tokens = 10


class _Resp:
    def __init__(self):
        msg = type("M", (), {"content": "```python\ndef transform(grid):\n    return grid\n```"})
        self.choices = [type("C", (), {"message": msg})]
        self.usage = _Usage()


class _Completions:
    def create(self, **kw):
        assert "messages" in kw and isinstance(kw["messages"][0]["content"], str)
        return _Resp()


class _FakeClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": _Completions()})()


# EndpointBackend normalises the url; no network until a call is made.
backend = EndpointBackend("http://localhost:9999", "my-model")
assert backend.base_url == "http://localhost:9999/v1"

orch = CloudOrchestrator(backend, selector="hemens",
                         num_tasks=3, num_retries=1, max_workers=2, budget=None)
fake = _FakeClient()
for r in ("writer", "coder"):
    orch._roles[r] = (fake, "my-model", "cloud")

# _complete must use the overridden chat-completions _raw_call.
txt = orch._complete("writer", [{"type": "text", "text": "hello"}], 128)
assert "transform" in txt, txt
print("cloud _complete via chat.completions OK:", repr(txt[:32]))

# Full run() against the fake endpoint (parent loop + cloud _raw_call).
res = orch.run()
assert res["num_tasks"] == 3
print("cloud run() OK:", res)


# ---- WallClockBudget caps on time ----
wb = WallClockBudget(gpu_cost_per_hour=3600.0, max_usd=0.001)  # $1 / second
time.sleep(0.02)
wb.record("my-model", 10, 2)
assert wb.would_exceed(), wb.summary()
print("WallClockBudget OK:", wb.summary())

print("ALL CLOUD OFFLINE TESTS PASSED")
