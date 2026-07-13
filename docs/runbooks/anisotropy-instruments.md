# Runbook — the anisotropy instruments (`measure_cone.py`, `probe_axis_poles.py`)

**Audience:** an operator or a *small agent LLM* driving these tools, plus a productization
sketch for reuse. **What they are:** two **read-only, offline** measurement scripts over a
*precomputed* embedding store. **No embedding model, no GPU, no network** — pure `numpy` over a
JSONL of `{chunk_id, embedding}`. They **REPORT NUMBERS ONLY** (no verdict); interpretation lives
in [`../axis-alignment.md`](../axis-alignment.md), [`../map.md`](../map.md), and
[`../research/co-adaptation-longitudinal.md`](../research/co-adaptation-longitudinal.md).

---

## 0. Preconditions

| need | detail |
|---|---|
| Python + `numpy` | run via the project venv: `./.venv/bin/python scripts/<name>.py` |
| the embedding store | `nv4096/corpus_4096.jsonl` (one JSON object per line: `chunk_id` str, `embedding` 4096 floats). Resolved **relative to repo root** unless absolute; a symlink target (`/mnt/storage/…`) is fine. |
| the text store (probe only) | `output/vectors/chunks.jsonl` — carries `speaker`, `source`, `conversation_name`, `timestamp`, `text`; joined by `chunk_id`. |
| RAM / time | ~1.4 GB for the (N≈87k × 4096) float32 matrix; a full run streams the 8 GB store + one `eigh` on a 4096×4096 matrix → a few minutes. |

Both scripts print **progress to stderr** and the **report to stdout** — so `> out.txt` captures a
clean report and an agent can parse stdout without progress noise. Exit codes: **2** = store not
found, **3** = no usable rows.

---

## 1. `measure_cone.py` — anisotropy diagnostics

**Run:**
```bash
./.venv/bin/python scripts/measure_cone.py
# options:
./.venv/bin/python scripts/measure_cone.py --corpus <path> --pairwise-sample 2000 --seed 0
```

**What it does:** streams the store with **dedup-keep-last** by `chunk_id` (later line wins — matches
`summarize_corpus.py`), **drops all-zero / wrong-shape rows** (the `_failed` markers), stacks the
survivors, and computes — all linear algebra in float64:
- counts (raw / distinct / dropped / N used) and a per-row L2-norm check (should be ≈1.0);
- `‖μ‖` and `‖μ‖²`; mean pairwise cosine on a random sample (sanity: ≈ `‖μ‖²`);
- isotropic baselines `1/√N`, `1/√4096`;
- **two eigenspectra** — UNCENTERED (`S = XᵀX/N`, cone included) and CENTERED (`C = XcᵀXc/N`, mean
  removed) — each with participation ratio, trace, top-1 fraction, cumulative variance at
  k∈{1,5,10,34,50,100}, and the first 10 eigenvalues.

**Read it:** de-meaning (uncentered→centered) is the first nullification step; the centered PR is the
effective rank after the cone offset is removed. Baseline reproduced 2026-07-12: ‖μ‖=0.555, centered
PR≈34, PR lifts 8→34 from de-meaning alone.

---

## 2. `probe_axis_poles.py` — what the top axes *are*

**Run:**
```bash
./.venv/bin/python scripts/probe_axis_poles.py
./.venv/bin/python scripts/probe_axis_poles.py --corpus <path> --chunks <path>
```

**What it does:** loads the store identically, computes the **centered covariance**, takes the **top-2
eigenvectors** (v0, v1), projects, and pulls the **pole passages** — top/bottom **12** for v0, **8**
for v1 — joining each `chunk_id` to its text via a **single streaming pass** over `chunks.jsonl`. It
also reports **composition** (speaker/source counts) for the top-200 vs bottom-200 by v0, which is how
you tell an axis is a *channel/register confound* rather than semantics.

**Read it:** each pole row is `[sigma=±X] speaker/source | conversation_name | text`. 2026-07-12
finding: v0 = export-channel, v1 = register; v0's −pole is a `[Tool use: Read]` duplication artifact.
That composition block is the tool-stub-hygiene alarm.

---

## 3. Operating notes (for a small agent driving these)

- **Deterministic** given the store + `--seed` (only the pairwise-cosine sample uses the seed).
- **Idempotent, side-effect-free** — they write nothing, mutate nothing; safe to re-run.
- **Store selection is the only real decision:** point `--corpus` at a *different* embedded store to
  measure a *different* population's cone. The dedup/zero-drop semantics are fixed and match the
  corpus-stats tooling, so numbers reconcile with `CORPUS_STATS.md`.
- **Failure surfaces are loud:** missing store → exit 2; empty after drop → exit 3; a missing text
  join is reported inline (`[MISSING JOINS]`), never silently skipped.

---

## 4. Productization sketch — toward reuse by our own tools and smaller agent LLMs

These are **research scripts today** (REPORTS-NUMBERS-ONLY, human-greppable stdout, `main()` interleaves
load+compute+print). To make them consumable by a small model or the wider system, in rising order of lift:

1. **Structured output.** Add a `--json` flag emitting a **stable schema** (the same numbers as one JSON
   object) so a small model consumes data, not prose. Keep the human report as the default.
2. **Factor a pure numeric core.** Split the linear algebra out of IO/CLI into an importable function
   over an `(N, d)` array — the same discipline as `alignment_core` (pure, exhaustively unit-testable
   with hand-built vectors). Load/print become thin shells. This is the ONE-DOOR-friendly shape.
3. **Parameterize the hardcoded knobs.** Pole counts (12/8), composition N (200), `CUM_K`, and the
   *top-2* axis count are literals — lift them to args for reuse on other stores/questions.
4. **Wrap as an MCP primitive** (`ARCHITECTURE.md` alignment). A stateless `measure_anisotropy` /
   `probe_axis` tool behind the one MCP door fits the contract — but note these read the **bulk store**,
   so they sit on the **data-plane / shared-substrate** side, not the analysis core; keep them from
   importing `mcp/commands/*`.
5. **The mood-classifier / SBR bridge (the payoff).** `measure_cone` already yields μ and the
   eigenbasis — the operator's *personal geometry*. A productized "personal state" tool would (a)
   **persist that basis as a calibration artifact** keyed by `model_name` (the `build_axis_null`
   manifest pattern — refuse a basis from the wrong embedder), and (b) expose **project-a-new-turn onto
   the axes** as the **State Vector** for SPEC-CATH-001's State-Based Resonance. Non-negotiable from the
   2026-07-12 finding: that projection must be **contrastive (z-scored vs the self-null)**, never
   absolute cosine, or SBR becomes the Emotional Echo Chamber the spec fears (R06).

**One-line summary for a small model:** *"These two scripts read a JSONL of embeddings and print the
shape of the cloud (measure_cone) and what its main axes mean (probe_axis_poles). No model, no GPU.
Point `--corpus` at a store; read stdout. To use programmatically, add `--json` and call the numeric
core directly."*
