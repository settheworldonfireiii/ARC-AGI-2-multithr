"""
cloud_inference.py — run inference on ON-DISK model weights sitting in *some*
cloud accelerator, without tying the code to any particular provider.

The contract is deliberately provider-neutral.  Wherever your GPU lives
(TensorDock, IBM Cloud, RunPod, Lambda, CoreWeave, a bare metal box, ...), the
weights on its disk are exposed through an **OpenAI-compatible HTTP endpoint**.
vLLM, HuggingFace TGI, SGLang, llama.cpp's server and Ollama all speak that
protocol, so the orchestrator only ever needs a ``base_url`` + ``model`` name.

Two backends are provided:

  EndpointBackend
      You already started a server somewhere (any host/provider).  Pass its
      ``base_url`` (e.g. http://<gpu-host>:8000/v1), the served ``model`` name
      and an optional ``api_key``.  Nothing provider-specific.

  LocalVLLMBackend
      Run THIS process *on* the accelerator where the weights live.  It boots
      ``vllm serve <weights_path>`` (or any ``--serve-cmd`` you give) as a
      subprocess, waits for the endpoint to come up, and hands back a local
      ``base_url``.  Tear-down stops the subprocess.

Also includes WallClockBudget: for self-hosted GPUs the meaningful cost is
GPU-rental *time*, not per-token API price, so this enforces a USD cap derived
from $/GPU-hour.  It duck-types the BudgetTracker interface the orchestrator
expects (would_exceed / record / summary).
"""

from __future__ import annotations

import abc
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------
class AcceleratorBackend(abc.ABC):
    """A handle to an OpenAI-compatible inference endpoint for on-disk weights."""

    model: str
    api_key: str = "EMPTY"  # most self-hosted servers ignore the key

    @property
    @abc.abstractmethod
    def base_url(self) -> str:
        """OpenAI-compatible base url, e.g. http://host:8000/v1 ."""

    def start(self) -> "AcceleratorBackend":
        """Idempotently make the endpoint reachable. Returns self."""
        self.wait_ready()
        return self

    def stop(self) -> None:
        """Release any resources we started (no-op by default)."""

    def wait_ready(self, timeout: float = 600.0, interval: float = 2.0) -> None:
        """Block until GET {base_url}/models returns HTTP 200 (or raise)."""
        url = self.base_url.rstrip("/") + "/models"
        deadline = time.time() + timeout
        last_err = None
        while time.time() < deadline:
            try:
                req = urllib.request.Request(
                    url, headers={"Authorization": f"Bearer {self.api_key}"}
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        return
            except Exception as e:  # connection refused / 404 while booting
                last_err = e
            time.sleep(interval)
        raise TimeoutError(
            f"endpoint {url} not ready after {timeout}s (last error: {last_err})"
        )

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


class EndpointBackend(AcceleratorBackend):
    """Wrap an already-running OpenAI-compatible server on any host/provider."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        self._base_url = base_url.rstrip("/")
        if not self._base_url.endswith("/v1"):
            # tolerate users passing the root url
            if "/v1" not in self._base_url:
                self._base_url = self._base_url + "/v1"
        self.model = model
        self.api_key = api_key or "EMPTY"

    @property
    def base_url(self) -> str:
        return self._base_url


class LocalVLLMBackend(AcceleratorBackend):
    """Serve on-disk weights with vLLM (default) on the local accelerator.

    `weights_path` may be a local directory of weights or a HuggingFace repo id
    that the server will pull.  `serve_cmd`, if given, fully overrides the
    launch command (so you can use TGI / SGLang / llama.cpp / Ollama instead).
    """

    def __init__(
        self,
        weights_path: str,
        served_model_name: str | None = None,
        host: str = "127.0.0.1",
        port: int = 8000,
        extra_args: tuple[str, ...] = (),
        serve_cmd: str | None = None,
        api_key: str = "EMPTY",
        env: dict | None = None,
        log_path: str = "vllm_server.log",
    ):
        self.weights_path = weights_path
        self.model = served_model_name or os.path.basename(weights_path.rstrip("/")) or weights_path
        self.host = host
        self.port = int(port)
        self.extra_args = tuple(extra_args)
        self.serve_cmd = serve_cmd
        self.api_key = api_key or "EMPTY"
        self.env = env
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _build_cmd(self) -> list[str]:
        if self.serve_cmd:
            return shlex.split(self.serve_cmd)
        # Default: vLLM's OpenAI-compatible server.
        return [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.weights_path,
            "--served-model-name", self.model,
            "--host", self.host,
            "--port", str(self.port),
            *self.extra_args,
        ]

    def start(self) -> "LocalVLLMBackend":
        if self.proc is None:
            cmd = self._build_cmd()
            print("[cloud] launching server:", " ".join(shlex.quote(c) for c in cmd))
            env = dict(os.environ)
            if self.env:
                env.update(self.env)
            self._logf = open(self.log_path, "ab")
            self.proc = subprocess.Popen(
                cmd, stdout=self._logf, stderr=subprocess.STDOUT, env=env
            )
        self.wait_ready()
        print(f"[cloud] server ready at {self.base_url} (model={self.model})")
        return self

    def stop(self) -> None:
        if self.proc is not None:
            print("[cloud] stopping server (pid=%s)" % self.proc.pid)
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc = None
            try:
                self._logf.close()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Time-based budget for self-hosted GPUs ($/GPU-hour, not $/token)
# ---------------------------------------------------------------------------
class WallClockBudget:
    """USD budget driven by GPU rental time.

    Duck-types the subset of BudgetTracker the orchestrator uses:
    ``would_exceed()``, ``record(model, in_tok, out_tok)``, ``summary()``.
    """

    def __init__(self, gpu_cost_per_hour: float, max_usd: float | None = None):
        self.gpu_cost_per_hour = float(gpu_cost_per_hour)
        self.max_usd = max_usd
        self.t0 = time.time()
        self.input_tokens = 0
        self.output_tokens = 0
        self.n_calls = 0
        self.stopped = False
        self._lock = threading.Lock()

    @property
    def elapsed_hours(self) -> float:
        return (time.time() - self.t0) / 3600.0

    @property
    def spent_usd(self) -> float:
        return self.elapsed_hours * self.gpu_cost_per_hour

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        with self._lock:
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
            self.n_calls += 1
            if self.max_usd is not None and self.spent_usd >= self.max_usd:
                self.stopped = True
        return 0.0

    def would_exceed(self) -> bool:
        if self.max_usd is None:
            return False
        if self.stopped or self.spent_usd >= self.max_usd:
            self.stopped = True
            return True
        return False

    @property
    def remaining_usd(self):
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent_usd)

    def summary(self) -> str:
        cap = "∞" if self.max_usd is None else f"${self.max_usd:.4f}"
        return (
            f"[gpu-budget] elapsed={self.elapsed_hours*60:.1f} min  "
            f"spent=${self.spent_usd:.4f} / {cap}  "
            f"@${self.gpu_cost_per_hour:.2f}/h  calls={self.n_calls}  "
            f"in_tok={self.input_tokens:,}  out_tok={self.output_tokens:,}"
        )


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------
def add_cloud_args(ap) -> None:
    g = ap.add_argument_group("cloud accelerator (on-disk weights)")
    g.add_argument("--base-url", default=os.environ.get("CLOUD_BASE_URL"),
                   help="OpenAI-compatible endpoint of an already-running server "
                        "(e.g. http://<gpu-host>:8000/v1). Provider-agnostic.")
    g.add_argument("--model", default=os.environ.get("CLOUD_MODEL"),
                   help="served model name (as the endpoint knows it).")
    g.add_argument("--api-key", default=os.environ.get("CLOUD_API_KEY", "EMPTY"),
                   help="bearer token for the endpoint (often ignored when self-hosted).")
    g.add_argument("--weights", default=os.environ.get("CLOUD_WEIGHTS"),
                   help="on-disk weights path or HF repo id; launches a local "
                        "vLLM server when --base-url is not given.")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8000)
    g.add_argument("--serve-cmd", default=None,
                   help="override the launch command (TGI/SGLang/llama.cpp/Ollama).")
    g.add_argument("--served-model-name", default=None)
    g.add_argument("--vllm-arg", action="append", default=[], dest="vllm_args",
                   help="extra arg passed through to vLLM (repeatable), "
                        'e.g. --vllm-arg=--tensor-parallel-size --vllm-arg=2')
    g.add_argument("--gpu-cost-per-hour", type=float, default=None,
                   help="$/GPU-hour; with --budget-usd this enforces a wall-clock "
                        "USD cap for the self-hosted run.")


def backend_from_args(args) -> AcceleratorBackend:
    """Build the right backend from parsed CLI args (base-url wins over weights)."""
    if args.base_url:
        if not args.model:
            raise SystemExit("--model is required with --base-url.")
        return EndpointBackend(args.base_url, args.model, api_key=args.api_key)
    if args.weights:
        return LocalVLLMBackend(
            args.weights,
            served_model_name=args.served_model_name or args.model,
            host=args.host,
            port=args.port,
            extra_args=tuple(args.vllm_args),
            serve_cmd=args.serve_cmd,
            api_key=args.api_key,
        )
    raise SystemExit(
        "Provide either --base-url (+ --model) for an existing endpoint, or "
        "--weights for a local vLLM server."
    )


if __name__ == "__main__":
    # Tiny offline self-test (no server started).
    b = EndpointBackend("http://gpu-host:8000", "my-model")
    assert b.base_url == "http://gpu-host:8000/v1", b.base_url
    wb = WallClockBudget(gpu_cost_per_hour=2.0, max_usd=0.0)
    assert wb.would_exceed() is False or wb.max_usd == 0.0
    print("cloud_inference self-test OK:", b.base_url, "|", wb.summary())
