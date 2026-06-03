# orch_2 inference upgrade

A separate, self-contained copy of the ARC-AGI-2 evolutionary self-refinement
orchestrator (`orch_2.py`, the file you saved in `Ivan_Radkevich_pm_env_slim/`),
with five changes to the **inference** path:

1. **Multithreaded** task solving.
2. **USD token budget** (CLI), with provider price tables and automatic
   derivation of `num_trials` / `context_length`.
3. A written **investigation of how compression and context addition work**
   (and fixes for two real bugs found there).
4. A **provider-agnostic cloud-accelerator version** (`orch_2_cloud.py`) that
   runs inference on on-disk model weights behind an OpenAI-compatible endpoint.
5. A **`--dataset` switch** adding a second domain — **AIME** competition math
   (popular, low solve-rate, exact integer grading) — next to ARC-AGI-2.

> Base file: I used the file literally named **`orch_2.py`** (latest, 30 May).
> Note there is also `orch_2_baseline.py` dated **14 May** in the source folder;
> they differ by ~64 lines (most visibly `compress_successful` gained a `code`
> argument). If you actually meant the 14 May `orch_2_baseline.py`, say so and
> I'll rebase onto it — the changes here are mechanical to re-apply.

---

## Folder contents

| File | What it is |
|------|------------|
| `orch_2.py`            | The orchestrator — **multithreaded + budget-aware**, with a real CLI. |
| `orch_2_cloud.py`      | Cloud-accelerator variant (`CloudOrchestrator`) for on-disk weights. |
| `pricing.py`           | Price tables, `BudgetTracker`, and the budget→(trials,context) resolver. |
| `cloud_inference.py`   | Provider-agnostic backends (endpoint / local vLLM) + `WallClockBudget`. |
| `math_dataset.py`      | AIME/math loading (HF datasets-server / file / offline sample) + answer grading. |
| `render.py`            | Grid→PNG rendering for the vision prompt (copied; one import fix, below). |
| `ARC-AGI-2/`           | The task data (1000 training tasks) the orchestrator reads. |
| `test_offline.py`      | Mocked end-to-end test (threads + budget + context fix), no network. |
| `test_cloud_offline.py`| Mocked test for the cloud path, no server/network. |
| `test_math_offline.py` | Mocked AIME run (grading + threads + no answer leak), no network. |
| `requirements.txt`     | Runtime dependencies. |

---

## Setup

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys come from the environment (NOT hardcoded — see Security note):
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...        # only if you use gpt/o* models
export FIREWORKS_API_KEY=fw-...     # only if you use Fireworks models
```

Run **from this folder** (the orchestrator computes the task dir as
`os.getcwd() + "/ARC-AGI-2/data/training/"`).

---

## Running the API orchestrator (`orch_2.py`)

```bash
# Defaults: writer=coder=claude-haiku-4-5, selector=hemens, 40 tasks, 8 threads.
python orch_2.py

# Pick models / concurrency:
python orch_2.py --writer claude-sonnet-4-6 --coder claude-haiku-4-5-20251001 \
                 --num-tasks 60 --workers 16

# Budget-driven (derive everything from $8):
python orch_2.py --budget-usd 8

# Budget + you fix the context length; trials are derived:
python orch_2.py --budget-usd 8 --context-length 4096

# Switch domain to AIME competition math (default source Maxwell-Jia/AIME_2024):
python orch_2.py --dataset aime --num-tasks 30
python orch_2.py --dataset aime --dataset-path AI-MO/aimo-validation-aime  # another HF set
python orch_2.py --dataset aime --dataset-path aime-sample                 # offline sample

python orch_2.py --help   # full list
```

Key flags: `--budget-usd`, `--num-trials`, `--context-length`/`--max-context`,
`--num-tasks`, `--workers`, `--price-per-1m-tokens`, `--price-file`, `--yes`.

---

## 1) Multithreading

`Orchestrator.run()` was fully sequential (a `for` loop per task, each making
several blocking LLM calls). The LLM calls are I/O-bound, so threads give a big
speed-up despite the GIL. The change:

* A reusable primitive, `Orchestrator._parallel_map(fn, items)`
  (`orch_2.py`), runs `fn` over items on a `ThreadPoolExecutor`
  (`--workers`, default 8), **preserving input order** and returning `None` for
  any item whose worker raised — so one task failing never sinks the batch.
* Three phases are parallelized:
  * **describe** every task (`write_description`),
  * **initial solve** (write code → exec → score → critique) across all tasks,
  * each **retry round** across the unsolved tasks in the curriculum cluster.
* Shared state is handled by having workers **return** results that the main
  thread applies (no shared-dict mutation inside workers). The one shared cache
  (`_succ_compress_cache`) and the `BudgetTracker` are guarded by locks.
* Cluster selection between rounds stays sequential (it's a sync point).

The Anthropic and OpenAI SDK clients are safe for concurrent use across threads.

`test_offline.py` asserts real parallelism (it observed **18** distinct worker
threads across the phases).

---

## 2) USD token budget

### Price tables (USD per 1M tokens, gathered June 2026)

In `pricing.py` (`PRICES`), matched by longest model-name prefix; editable, or
override with `--price-file` / `--price-per-1m-tokens`.

| Provider | Model | input | output |
|---|---|---:|---:|
| Anthropic | claude-haiku-4-5 | 1.00 | 5.00 |
| Anthropic | claude-sonnet-4-6 | 3.00 | 15.00 |
| Anthropic | claude-opus-4-7 | 5.00 | 25.00 |
| OpenAI | gpt-4o | 2.50 | 10.00 |
| OpenAI | gpt-4.1 | 2.00 | 8.00 |
| OpenAI | o3 | 2.00 | 8.00 |
| OpenAI | gpt-5 | 1.25 | 10.00 |
| OpenAI | gpt-4o-mini | 0.15 | 0.60 |
| Fireworks | < 16B (e.g. llama-8b) | 0.20 | 0.20 |
| Fireworks | > 16B (e.g. llama-70b) | 0.90 | 0.90 |
| Fireworks | deepseek-v3 | 0.56 | 1.68 |

Sources: Anthropic <https://platform.claude.com/docs/en/about-claude/pricing>,
OpenAI <https://openai.com/api/pricing/>,
Fireworks <https://fireworks.ai/pricing>. **Prices change — verify before relying on them.**

### How the budget maps to run parameters

The planning identity (your formula):

```
num_trials * context_length * num_tasks  ==  budget_tokens
budget_tokens = budget_usd / price_per_token
price_per_token = blended_$per_Mtok / 1e6
```

`blended_$per_Mtok` defaults to the average of the writer model's input &
output price (`--io-input-weight` tunes the input share; ARC prompts are
input-heavy). Override the whole thing with `--price-per-1m-tokens`.

`resolve_budget()` in `pricing.py` applies exactly the rules you asked for:

| You pass | Result |
|---|---|
| no `--budget-usd` | budget **off**; use given/default trials & context. |
| budget only | keep default context, **derive** `num_trials` (or shrink context if the budget is tiny). |
| budget + `--context-length` | **derive** `num_trials`. |
| budget + `--num-trials` | **derive** `context_length`. |
| budget + **both**, consistent (±15%) | use them; budget stays a hard runtime cap. |
| budget + **both**, inconsistent | **warn**, print a **balanced** proposal keeping `context_length` as the basis and recomputing `num_trials`. `--yes` (or typing `y`) ⇒ apply it; **`n` ⇒ disable the budget and run your `num_trials`/`context_length` unchanged.** |

`context_length` is enforced as the **max generated (output) tokens per call**
(`max_output_tokens`), so it is a real cost lever, not just bookkeeping.

### Runtime enforcement

`BudgetTracker` (thread-safe) meters **actual** spend from each response's token
usage (`response.usage`), across all three providers. Before every call the
orchestrator checks `would_exceed()`; once the cap is hit it stops scheduling new
work and prints a summary like:

```
[budget] spent=$0.0105 / $0.0100  calls=5  in_tok=6,000  out_tok=900
```

(A handful of in-flight calls may finish after the cap is crossed; that overshoot
is bounded by `--workers`.)

---

## 3) How compression & context addition work (investigation)

This was a request to **check** the mechanism. Findings, with fixes applied where
they were outright broken:

**Compression functions** (`orch_2.py`):

* `compress_successful(desc, code)` — asks the writer to summarise a solved
  task's rule + code to ≤200–400 tokens. It **is** used: each retry round folds
  the solved cluster members into a `succ_acc` block that is appended to the
  prompt of the tasks being retried. *Issue found:* it was re-compressing the
  **same** success on **every** round (redundant paid calls). **Fix:** results
  are now cached per task in `_succ_compress_cache` (computed once, in parallel).
* `compress_criticism(desc)` — defined to summarise failed-attempt criticism to
  ≤150–250 tokens, but it has **zero call sites**. So criticism is **never
  compressed**: the critique chain is appended raw and grows every round, while
  successes are compressed. Left as-is (behavioural), but flagged — wiring it in
  is the obvious next step if the critique context gets too long.

**Context addition** — the curriculum retry loop builds each retry prompt by
appending: the task, the formatted task text, the raw blocks, the JSON task, the
accumulated criticism, and the compressed successful examples.

* **Bug (now fixed):** the original did
  `prompt += task` / `prompt += fb` / `prompt += json_task`, where `task` and
  `json_task` are pydantic `Challenge` objects and `fb` is a `list[dict]`.
  `str + Challenge` / `str + list` raises `TypeError`, so **the moment the retry
  phase touched any unsolved task, `run()` crashed.** Every helper that coerces
  context now goes through `Orchestrator._as_text()` (Challenge →
  `model_dump_json()`, list → `json.dumps`, str → itself, None → ""). The retry
  loop runs to completion (validated in `test_offline.py`).
* `re_describe()` — a cleaner prompt builder that interleaves critique +
  successful examples — exists but its only call was **commented out**; the
  active path is the raw `prompt += …` accumulation above. It is now unreferenced
  (dead code); kept for reference.
* **Note (unchanged):** retry prompts accumulate across rounds (each round stores
  the grown prompt and appends again), so *input* context grows over rounds. The
  new `context_length` cap limits *generation*, and the budget meters total
  spend, but the input-growth design is preserved.

**Other robustness fixes made along the way:**

* `select_cluster` could raise `ValueError: zero-size array` when a cluster is
  all-solved/all-unsolved; `run()` now calls it via `_cluster_or_random()`, which
  falls back to a random cluster instead of killing the run.
* `write_code` looped `while not content:` — an infinite loop if the model kept
  failing. It's now a bounded 3-try loop.
* Fireworks was constructed as `fireworks.Fireworks(...)`, which doesn't exist in
  the installed `fireworks` package; it's now routed through the OpenAI-compatible
  Fireworks endpoint, so Fireworks models actually work.
* `render.py` imported `color_map` from `orch_2` (a circular import that only
  worked when `orch_2.py` was run as `__main__`, and made `import orch_2` fail).
  `render.py` redefines `color_map` immediately anyway, so that import was
  removed — now `orch_2` is importable as a module (required by `orch_2_cloud.py`).

---

## 4) Cloud accelerator on on-disk weights (`orch_2_cloud.py`)

Not tied to any provider. The contract: weights live on disk in *some*
accelerator (TensorDock, IBM Cloud, RunPod, Lambda, bare metal, …) and are
exposed through an **OpenAI-compatible HTTP endpoint** (vLLM / TGI / SGLang /
llama.cpp / Ollama all speak it). `CloudOrchestrator` subclasses the normal
orchestrator — keeping the multithreading, curriculum loop and budget — and
points every role at that one endpoint, talking to it via **Chat Completions**
(the universal compatible API), with prompts flattened to text.

`cloud_inference.py` provides two backends:

* **`EndpointBackend`** — you already started a server somewhere; pass
  `--base-url http://<gpu-host>:8000/v1 --model <name>`.
* **`LocalVLLMBackend`** — run `orch_2_cloud.py` **on** the accelerator; it boots
  `vllm serve <weights>` (or any `--serve-cmd`), waits for `/v1/models`, then
  runs. `--vllm-arg` passes flags through (e.g. tensor-parallel).

```bash
# A) existing endpoint on any host/provider:
python orch_2_cloud.py --base-url http://<gpu-host>:8000/v1 --model my-llama \
        --num-tasks 40 --workers 8

# B) boot vLLM here on local weights, 2 GPUs, with a $/GPU-hour budget cap:
python orch_2_cloud.py --weights /data/models/Qwen2.5-32B \
        --vllm-arg=--tensor-parallel-size --vllm-arg=2 \
        --gpu-cost-per-hour 3.20 --budget-usd 5 --num-tasks 60
```

**Budget for self-hosted GPUs:** per-token pricing doesn't apply, so:

* `--gpu-cost-per-hour R` + `--budget-usd B` ⇒ `WallClockBudget` stops the run
  after `B/R` hours (time is the cap; trials/context are taken as given).
* `--price-per-1m-tokens P` ⇒ behaves like the API budget (amortised token price).
* neither ⇒ no cap.

---

## 5) Switchable datasets — ARC grids + AIME math (`--dataset`)

`--dataset arc` (default) is unchanged. `--dataset aime` runs the **same**
multithreaded + budgeted + curriculum loop on **AIME** competition math.

Why AIME and not IMO proper: IMO is proof-based and can't be auto-graded by exact
match. AIME is popular, has a famously low model solve rate, and every answer is a
single integer 0–999 — so it grades exactly, matching this orchestrator's
"score = fraction correct" design.

How it works (`MathDomainMixin` in `orch_2.py`, plus `math_dataset.py`):

* **Load** — `--dataset-path` takes a local `.json`/`.jsonl` **or** a HuggingFace
  dataset id, fetched through the public datasets-server API (just `urllib`; no
  heavy `datasets` dependency). Default source for `--dataset aime` is
  `Maxwell-Jia/AIME_2024`; `aime-sample` is a small bundled offline set.
* **Solve** — describe → *plan a strategy*; write_code → *produce a full solution
  ending in `\boxed{answer}`*; run_code → *extract the boxed answer*;
  parse_output → *grade* (int / float / sympy / string compare).
* **No cheating** — the task object carries the gold answer for grading, but the
  mixin's `_as_text` strips it before any task is folded into a prompt, and
  `write_self_critique` never echoes it (verified in `test_math_offline.py`).
* Everything else (threads, USD budget, critique, compression, curriculum
  clustering with math-flavoured tags) is inherited unchanged. The cloud path has
  it too: `MathCloudOrchestrator` = math domain + on-disk-weights endpoint.

```bash
python orch_2.py --dataset aime --num-tasks 30 --budget-usd 5
python orch_2.py --dataset aime --dataset-path AI-MO/aimo-validation-aime
python orch_2.py --dataset aime --dataset-path ./my_problems.jsonl   # {id,problem,answer}
python orch_2_cloud.py --base-url http://<host>:8000/v1 --model m --dataset aime
```

Add another benchmark by pointing `--dataset-path` at any `{problem, answer}`
JSONL (MATH, OlympiadBench, Omni-MATH, …) — no code changes needed.

---

## Security note

The original `__main__` had a **live `ANTHROPIC_API_KEY` hardcoded** (it also
appears in `orch_2_acc.py`). This copy reads keys from the environment instead.
**Rotate that key** — committed/shared secrets should be considered compromised.

## Verify without spending tokens

```bash
python orch_2.py --help
python orch_2_cloud.py --help
python pricing.py            # budget-resolver demo
python math_dataset.py       # answer extraction + grading self-test
python test_offline.py       # threads + budget + context fix (mocked)
python test_cloud_offline.py # cloud path (mocked)
python test_math_offline.py  # AIME domain: grading + threads + no answer leak (mocked)
```
