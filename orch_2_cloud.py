"""
orch_2_cloud.py — the ARC-AGI-2 orchestrator running against ON-DISK model
weights in a cloud accelerator instead of a hosted API.

It subclasses orch_2.Orchestrator (so it keeps the multithreading, the
curriculum self-refinement loop and the budget machinery) and points every
role (writer / coder / selector) at one OpenAI-compatible endpoint served from
the on-disk weights.  The endpoint is provider-agnostic — see cloud_inference.py
(EndpointBackend for an already-running server anywhere, or LocalVLLMBackend to
boot vLLM on the accelerator where the weights live).

Examples
--------
# A) talk to an endpoint you already started on some GPU host (TensorDock,
#    IBM Cloud, RunPod, bare metal — doesn't matter):
python orch_2_cloud.py --base-url http://<gpu-host>:8000/v1 \
        --model my-llama --num-tasks 40 --workers 8

# B) run ON the accelerator; boot vLLM on local on-disk weights, 2 GPUs,
#    with a $/GPU-hour budget cap of $5:
python orch_2_cloud.py --weights /data/models/Qwen2.5-32B \
        --vllm-arg=--tensor-parallel-size --vllm-arg=2 \
        --gpu-cost-per-hour 3.20 --budget-usd 5 --num-tasks 60
"""

from __future__ import annotations

import argparse
import os

import openai

import orch_2
from orch_2 import Orchestrator, MathDomainMixin, _usage_tokens
import pricing
from pricing import BudgetTracker, resolve_budget
import cloud_inference
from cloud_inference import WallClockBudget, add_cloud_args, backend_from_args


class CloudOrchestrator(Orchestrator):
    """Orchestrator whose writer/coder/selector all hit one local-weights endpoint."""

    def __init__(self, backend: cloud_inference.AcceleratorBackend,
                 selector: str = "hemens", **kw):
        self.backend = backend
        model = backend.model

        # Initialise the parent with the served model as every role and a dummy
        # key (the throwaway clients it builds are immediately overridden below).
        super().__init__(
            model, model, selector,
            "EMPTY", "EMPTY", "EMPTY" if selector != "hemens" else "hemens",
            **kw,
        )

        # One OpenAI-compatible client pointed at the on-disk-weights endpoint.
        client = openai.OpenAI(base_url=backend.base_url, api_key=backend.api_key or "EMPTY")
        self._roles["writer"] = (client, model, "cloud")
        self._roles["coder"] = (client, model, "cloud")
        if selector != "hemens":
            self._roles["selector"] = (client, model, "cloud")
            self.selector = client
            self.selectormodel = model

        # Keep the backwards-compatible attributes consistent.
        self.writer, self.writermodel, self._writer_provider = client, model, "cloud"
        self.coder, self.codermodel, self._coder_provider = client, model, "cloud"

    @staticmethod
    def _to_text(content) -> str:
        """Flatten a str / list-of-content-blocks into plain text.

        Self-hosted models may not be multimodal, and not every OpenAI-compatible
        server accepts structured content parts, so we send text only (image
        blocks, if any, are dropped)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for b in content:
                if isinstance(b, dict) and b.get("type") == "text":
                    parts.append(b.get("text", ""))
            return "\n".join(parts)
        return str(content)

    def _raw_call(self, client, model, provider, content, max_tokens):
        """Always use Chat Completions — the universal OpenAI-compatible API."""
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": self._to_text(content)}],
            max_tokens=max_tokens,
        )
        text = resp.choices[0].message.content
        in_tok, out_tok = _usage_tokens(resp)
        return text, in_tok, out_tok


class MathCloudOrchestrator(MathDomainMixin, CloudOrchestrator):
    """AIME-style math on cloud on-disk weights: math domain + chat endpoint."""
    pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="ARC-AGI-2 orchestrator on cloud-accelerator on-disk weights "
                    "(multithreaded, budget-aware, provider-agnostic).")
    ap.add_argument("--selector", default="hemens",
                    help='"hemens" (uses the served model for tagging) or "model".')
    ap.add_argument("--dataset", default="arc", choices=["arc", "aime"],
                    help="task domain: 'arc' (grids, default) or 'aime' (math).")
    ap.add_argument("--dataset-path", default=None,
                    help="math only: local .json/.jsonl file or HF dataset id "
                         "(or 'aime-sample' for the offline sample).")
    ap.add_argument("--num-tasks", type=int, default=40)
    ap.add_argument("--num-trials", type=int, default=None)
    ap.add_argument("--context-length", "--max-context", dest="context_length",
                    type=int, default=None,
                    help="max generated tokens per call.")
    ap.add_argument("--workers", type=int, default=8)
    # Budget: token-amortised (--price-per-1m-tokens) OR time-based (--gpu-cost-per-hour)
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--price-per-1m-tokens", type=float, default=None,
                    help="amortised $/1M-tokens for the self-hosted model; enables "
                         "a token-based budget like the API orchestrator.")
    ap.add_argument("--io-input-weight", type=float, default=0.5)
    ap.add_argument("--yes", action="store_true")
    add_cloud_args(ap)
    return ap


def _plan_budget(args, model):
    """Return (num_trials, context_length, budget_obj). Two cost models:

      * --gpu-cost-per-hour set  -> WallClockBudget (time is the cap; trials &
        context are taken as-is, NOT derived, since tokens are ~free).
      * --price-per-1m-tokens set -> token BudgetTracker via resolve_budget.
      * neither -> no budget; use given/default trials & context.
    """
    default_trials = 4
    default_ctx = 8192

    if args.gpu_cost_per_hour is not None:
        num_trials = args.num_trials if args.num_trials is not None else default_trials
        context_length = args.context_length if args.context_length is not None else default_ctx
        budget = WallClockBudget(args.gpu_cost_per_hour, max_usd=args.budget_usd)
        print(f"[budget] GPU-hour budget: ${args.budget_usd} cap @ "
              f"${args.gpu_cost_per_hour}/h; trials={num_trials} context={context_length} "
              "(not derived from budget — time is the cap).")
        return num_trials, context_length, budget

    if args.budget_usd is not None and args.price_per_1m_tokens is not None:
        # Make the runtime tracker price this model at the amortised rate.
        pricing.PRICES[model] = (args.price_per_1m_tokens, args.price_per_1m_tokens)
        plan = resolve_budget(
            budget_usd=args.budget_usd,
            num_trials=args.num_trials,
            context_length=args.context_length,
            num_tasks=args.num_tasks,
            blended_mtok=args.price_per_1m_tokens,
            assume_yes=args.yes,
        )
        for note in plan.notes:
            print("[budget]", note)
        budget = BudgetTracker(max_usd=plan.max_usd) if plan.budget_enabled else None
        return plan.num_trials, plan.context_length, budget

    num_trials = args.num_trials if args.num_trials is not None else default_trials
    context_length = args.context_length if args.context_length is not None else default_ctx
    if args.budget_usd is not None:
        print("[budget] --budget-usd given without --price-per-1m-tokens or "
              "--gpu-cost-per-hour; cannot map a self-hosted budget to tokens. "
              "Running without a budget cap.")
    return num_trials, context_length, None


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    backend = backend_from_args(args)
    backend.start()  # waits until the endpoint is reachable (or launches vLLM)
    model = backend.model

    num_trials, context_length, budget = _plan_budget(args, model)
    print(f"[cloud] model={model} endpoint={backend.base_url} "
          f"trials={num_trials} context={context_length} workers={args.workers}")

    try:
        cls = CloudOrchestrator if args.dataset == "arc" else MathCloudOrchestrator
        orch = cls(
            backend,
            selector=args.selector,
            num_tasks=args.num_tasks,
            num_retries=num_trials,
            max_workers=args.workers,
            max_output_tokens=context_length,
            budget=budget,
        )
        if args.dataset != "arc":
            orch.dataset_source = args.dataset
            orch.dataset_path = args.dataset_path
        print(f"[dataset] {args.dataset}"
              + (f" (source={args.dataset_path})" if args.dataset_path else ""))
        orch.run()
    finally:
        backend.stop()


if __name__ == "__main__":
    main()
