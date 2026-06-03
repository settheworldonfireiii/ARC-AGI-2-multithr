"""
pricing.py — token price tables + USD budget machinery for orch_2.

Prices are USD per 1,000,000 tokens, gathered ~June 2026 from the public
pricing pages of each provider.  THEY CHANGE OFTEN — override at runtime with
``--price-per-1m-tokens`` / ``--price-file`` or just edit ``PRICES`` below.

Sources (June 2026):
  Anthropic : https://platform.claude.com/docs/en/about-claude/pricing
  OpenAI    : https://openai.com/api/pricing/  (and developers.openai.com/api/docs/pricing)
  Fireworks : https://fireworks.ai/pricing   (and docs.fireworks.ai/serverless/pricing)

This module is intentionally dependency-free (stdlib only) so it can be unit
tested and reused by both the API orchestrator (orch_2.py) and the
cloud-accelerator orchestrator (orch_2_cloud.py).
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Price table: model-name (or prefix) -> (input_$_per_Mtok, output_$_per_Mtok)
# ---------------------------------------------------------------------------
# Keys are matched against the *start* of the model id, longest key first, so
# "claude-haiku-4-5" matches "claude-haiku-4-5-20251001".  Add/adjust freely.
PRICES: dict[str, tuple[float, float]] = {
    # ---- Anthropic (per 1M tokens) -------------------------------------
    "claude-opus-4-7":    (5.00, 25.00),
    "claude-opus-4-6":    (5.00, 25.00),
    "claude-opus":        (5.00, 25.00),
    "claude-sonnet-4-6":  (3.00, 15.00),
    "claude-sonnet":      (3.00, 15.00),
    "claude-haiku-4-5":   (1.00,  5.00),
    "claude-haiku-3-5":   (0.80,  4.00),
    "claude-haiku":       (1.00,  5.00),
    # ---- OpenAI --------------------------------------------------------
    "gpt-5":              (1.25, 10.00),
    "gpt-4.1-mini":       (0.40,  1.60),
    "gpt-4.1":            (2.00,  8.00),
    "gpt-4o-mini":        (0.15,  0.60),
    "gpt-4o":             (2.50, 10.00),
    "o3":                 (2.00,  8.00),
    "gpt":                (2.50, 10.00),   # generic gpt fallback
    # ---- Fireworks serverless (open-weight models) ---------------------
    # Fireworks bills by model size band; these are representative June-2026
    # serverless rates.  "deepseek-v3" listed explicitly; otherwise the
    # size-band fallbacks below are used.
    "deepseek-v3":        (0.56,  1.68),
    "deepseek":           (0.56,  1.68),
    "llama-v3p1-70b":     (0.90,  0.90),
    "llama-v3p1-8b":      (0.20,  0.20),
    "qwen":               (0.90,  0.90),
    "fireworks-small":    (0.20,  0.20),   # < 16B params
    "fireworks-large":    (0.90,  0.90),   # > 16B params
}

# Used when a model id matches nothing above.  Conservative-ish (haiku-like).
DEFAULT_PRICE: tuple[float, float] = (1.00, 5.00)

# Self-hosted / on-disk-weights inference has no per-token API charge.  The
# cloud orchestrator registers this so the per-token budget does not block it;
# real cost there is GPU-rental per hour (see orch_2_cloud.py --gpu-cost-per-hour).
SELF_HOSTED_PRICE: tuple[float, float] = (0.0, 0.0)


def load_price_file(path: str) -> None:
    """Merge a JSON file of {model: [input_per_mtok, output_per_mtok]} into PRICES."""
    with open(path) as f:
        data = json.load(f)
    for k, v in data.items():
        PRICES[k] = (float(v[0]), float(v[1]))


def price_for(model: str) -> tuple[float, float]:
    """Return (input_$_per_Mtok, output_$_per_Mtok) for *model*.

    Matches the longest registered key that is a prefix of ``model`` so that
    versioned ids (e.g. ``claude-haiku-4-5-20251001``) resolve correctly.
    """
    if model in PRICES:
        return PRICES[model]
    best = None
    for key in PRICES:
        if model.startswith(key):
            if best is None or len(key) > len(best):
                best = key
    return PRICES[best] if best is not None else DEFAULT_PRICE


def blended_price_per_mtok(model: str, input_weight: float = 0.5) -> float:
    """Single blended $/Mtok for planning math.

    ``input_weight`` is the fraction of a 'token unit' assumed to be input
    (default 0.5 => simple average of input & output price).  ARC-AGI prompts
    are input-heavy, so 0.5 is a reasonable middle; tune with --io-input-weight.
    """
    in_p, out_p = price_for(model)
    return input_weight * in_p + (1.0 - input_weight) * out_p


# ---------------------------------------------------------------------------
# Runtime budget tracking (thread-safe — the orchestrator is multithreaded)
# ---------------------------------------------------------------------------
class BudgetExceeded(RuntimeError):
    """Raised by the orchestrator when a call would exceed the USD budget."""


@dataclass
class BudgetTracker:
    """Accumulates real USD spend from per-call token usage.

    ``max_usd is None`` means 'no cap' (tracking only — never blocks).
    All mutating access is guarded by a lock so worker threads can record
    spend concurrently.
    """

    max_usd: float | None = None
    spent_usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    n_calls: int = 0
    # stop-gap so a burst of in-flight threads can't blow far past the cap:
    # once spent crosses max_usd we set this and the orchestrator stops
    # scheduling new work.
    stopped: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_p, out_p = price_for(model)
        cost = (input_tokens / 1e6) * in_p + (output_tokens / 1e6) * out_p
        with self._lock:
            self.spent_usd += cost
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)
            self.n_calls += 1
            if self.max_usd is not None and self.spent_usd >= self.max_usd:
                self.stopped = True
        return cost

    def would_exceed(self) -> bool:
        """True if the cap is already reached (cheap, lock-free read)."""
        if self.max_usd is None:
            return False
        return self.stopped or self.spent_usd >= self.max_usd

    @property
    def remaining_usd(self) -> float | None:
        if self.max_usd is None:
            return None
        return max(0.0, self.max_usd - self.spent_usd)

    def summary(self) -> str:
        cap = "∞" if self.max_usd is None else f"${self.max_usd:.4f}"
        return (
            f"[budget] spent=${self.spent_usd:.4f} / {cap}  "
            f"calls={self.n_calls}  in_tok={self.input_tokens:,}  "
            f"out_tok={self.output_tokens:,}"
        )


# ---------------------------------------------------------------------------
# Budget <-> (num_trials, context_length) resolution  (task #2 of the request)
# ---------------------------------------------------------------------------
@dataclass
class BudgetPlan:
    num_trials: int
    context_length: int
    budget_enabled: bool
    max_usd: float | None
    notes: list[str] = field(default_factory=list)


def _tokens_for_budget(budget_usd: float, blended_mtok: float) -> float:
    """How many (blended) tokens the USD budget buys."""
    if blended_mtok <= 0:
        return float("inf")  # self-hosted / free => effectively unlimited tokens
    return budget_usd / (blended_mtok / 1e6)


def resolve_budget(
    *,
    budget_usd: float | None,
    num_trials: int | None,
    context_length: int | None,
    num_tasks: int,
    blended_mtok: float,
    default_context_length: int = 8192,
    default_num_trials: int = 4,
    min_context_length: int = 256,
    max_context_length: int = 32768,
    tolerance: float = 0.15,
    assume_yes: bool = False,
    prompter=input,
    log=print,
) -> BudgetPlan:
    """Reconcile a USD budget with num_trials / context_length per the spec.

    Planning identity (the user's formula):

        num_trials * context_length * num_tasks == budget_tokens

    where ``budget_tokens = budget_usd / price_per_token`` and
    ``price_per_token = blended_mtok / 1e6``.

    Behaviour:
      * No budget                  -> budget disabled; keep given/default trials & context.
      * Budget + neither           -> keep a sane default context, derive trials
                                       (fall back to deriving context if trials < 1).
      * Budget + exactly one       -> derive the other from the identity.
      * Budget + BOTH, consistent  -> use them (budget stays enabled as a hard cap).
      * Budget + BOTH, inconsistent-> warn; offer a *balanced* plan that keeps
                                       context_length as the basis and recomputes
                                       num_trials.  yes => apply it; no => DISABLE
                                       the budget and enforce the user's trials &
                                       context unchanged.
    """
    notes: list[str] = []

    # ---- (a) no budget -------------------------------------------------
    if budget_usd is None:
        return BudgetPlan(
            num_trials=num_trials if num_trials is not None else default_num_trials,
            context_length=context_length if context_length is not None else default_context_length,
            budget_enabled=False,
            max_usd=None,
            notes=["No --budget-usd given; budget tracking disabled."],
        )

    budget_tokens = _tokens_for_budget(budget_usd, blended_mtok)
    notes.append(
        f"Budget ${budget_usd:.4f} ≈ {budget_tokens:,.0f} blended tokens "
        f"at ${blended_mtok:.2f}/Mtok over {num_tasks} tasks."
    )

    def derive_trials(ctx: int) -> int:
        if ctx <= 0 or num_tasks <= 0:
            return default_num_trials
        return int(budget_tokens // (ctx * num_tasks))

    def derive_context(tr: int) -> int:
        if tr <= 0 or num_tasks <= 0:
            return default_context_length
        return int(budget_tokens // (tr * num_tasks))

    def clamp_ctx(ctx: int) -> int:
        c = max(min_context_length, min(max_context_length, ctx))
        if c != ctx:
            notes.append(f"context_length clamped {ctx} -> {c} "
                         f"[{min_context_length}, {max_context_length}].")
        return c

    have_tr = num_trials is not None
    have_ctx = context_length is not None

    # ---- (b) budget only ----------------------------------------------
    if not have_tr and not have_ctx:
        ctx = default_context_length
        tr = derive_trials(ctx)
        if tr < 1:
            # context default already eats the whole budget — keep 1 trial,
            # shrink context instead.
            tr = 1
            ctx = clamp_ctx(derive_context(tr))
            notes.append("Budget too small for default context; set trials=1 and "
                         f"shrank context to {ctx}.")
        else:
            notes.append(f"Derived num_trials={tr} from budget "
                         f"(context_length={ctx} default).")
        return BudgetPlan(tr, ctx, True, budget_usd, notes)

    # ---- (c) budget + exactly one -------------------------------------
    if have_ctx and not have_tr:
        ctx = clamp_ctx(int(context_length))
        tr = max(1, derive_trials(ctx))
        notes.append(f"Derived num_trials={tr} from budget and your "
                     f"context_length={ctx}.")
        return BudgetPlan(tr, ctx, True, budget_usd, notes)

    if have_tr and not have_ctx:
        tr = max(1, int(num_trials))
        ctx = clamp_ctx(derive_context(tr))
        notes.append(f"Derived context_length={ctx} from budget and your "
                     f"num_trials={tr}.")
        return BudgetPlan(tr, ctx, True, budget_usd, notes)

    # ---- (d) budget + BOTH --------------------------------------------
    tr = int(num_trials)        # type: ignore[arg-type]
    ctx = int(context_length)   # type: ignore[arg-type]
    requested = tr * ctx * num_tasks
    ratio = requested / budget_tokens if budget_tokens > 0 else float("inf")

    if abs(ratio - 1.0) <= tolerance:
        notes.append(
            f"All three set and consistent within {tolerance:.0%} "
            f"(requested {requested:,.0f} vs budget {budget_tokens:,.0f} tokens)."
        )
        return BudgetPlan(tr, ctx, True, budget_usd, notes)

    # Inconsistent -> warn + offer balanced plan (context_length is the basis).
    balanced_trials = max(1, round(budget_tokens / (ctx * num_tasks)))
    over_under = "OVER" if requested > budget_tokens else "UNDER"
    est_usd = requested / budget_tokens * budget_usd if budget_tokens > 0 else float("inf")

    log("")
    log("=" * 72)
    log("BUDGET WARNING: num_trials × context_length × num_tasks does not match "
        "the USD budget.")
    log(f"  num_tasks       = {num_tasks}")
    log(f"  context_length  = {ctx}      (kept as the basis for balancing)")
    log(f"  num_trials      = {tr}")
    log(f"  -> requested    = {requested:,.0f} tokens  (~${est_usd:.4f}, {over_under} budget)")
    log(f"  budget          = ${budget_usd:.4f} ≈ {budget_tokens:,.0f} tokens")
    log("")
    log(f"  Balanced proposal (keep context_length={ctx}, num_tasks={num_tasks}):")
    log(f"      num_trials: {tr}  ->  {balanced_trials}")
    log("=" * 72)

    if assume_yes:
        accept = True
        log("[--yes] auto-accepting balanced proposal.")
    elif not sys.stdin.isatty():
        # Non-interactive (CI, nohup, piped stdin): never block on input().
        # Safe default == the spec's "no" branch: disable budget, keep values.
        accept = False
        log("Non-interactive (no TTY) and no --yes: declining balanced plan.")
    else:
        try:
            ans = prompter("Apply the balanced plan? [y/N]: ").strip().lower()
        except EOFError:
            ans = ""
        accept = ans in ("y", "yes")

    if accept:
        notes.append(f"User accepted balanced plan: num_trials {tr} -> {balanced_trials} "
                     f"(context_length={ctx} kept).")
        return BudgetPlan(balanced_trials, ctx, True, budget_usd, notes)
    else:
        notes.append("User declined balanced plan: budget DISABLED; enforcing your "
                     f"num_trials={tr} and context_length={ctx} as given.")
        return BudgetPlan(tr, ctx, False, None, notes)


if __name__ == "__main__":
    # Tiny self-test / demo of the resolver (no API calls).
    import sys
    bp = blended_price_per_mtok("claude-haiku-4-5-20251001")
    print(f"haiku blended $/Mtok = {bp:.2f}")
    plan = resolve_budget(
        budget_usd=10.0, num_trials=10, context_length=8192, num_tasks=40,
        blended_mtok=bp, assume_yes=True,
    )
    print(plan)
    sys.exit(0)
