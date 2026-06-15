# ADR-SKMCP-0002: Bearing-Analysis Tool Contract for sk-mcp

**Status:** proposed
**Date:** 2026-06-15 (US/Mountain)
**Author:** shanevcantwell, with Claude (Claude Code) as drafting collaborator
**Related:** ADR-SKMCP-0001 (the directional-projection *primitive* this contract operationalizes); ADR-001 (the position-regime sibling whose `build_axis_null.py` → `analyze_axis_alignment` pair is the template); ADR-002 (canonical `model_name`, sk-mcp #11); ADR-003 (stateless MCP cutover, sk-mcp #2); `data/anchors/escalation_grid.yaml` (the first axis input); `docs/HANDOFF.md` §8 (regime split)

**Supersedes:** —
**Superseded by:** —

> **Relationship to ADR-SKMCP-0001:** 0001 fixes the *math* — what a directional
> projection is (signed component + orthogonal residual + cosine), that the null
> must be measured, and that a measurement is single-embedder. This ADR (0002)
> fixes the *tool contract* — how that primitive is surfaced as an MCP tool family
> an agent can drive, what artifact it builds, and what misuse it refuses. 0001 is
> the instrument; 0002 is how it ships.

---

## Context

ADR-SKMCP-0001 specified the bearing primitive but left the tool surface open. Three
facts constrain how it should ship, and together they pick the shape:

1. **The confidence procedure is multi-step, but the north star is one-step.** A
   trustworthy bearing measurement requires: validate that the axis is even coherent
   (the union-SVD / coherence check — "Spike A"), build a *measured* displacement null
   under anisotropy, and only then project. The standing project rule is that no matter
   how many steps it takes to become confident, *we are not done until that whole
   process is reproducible as one action.* The human (or agent) must not be the
   integration layer.

2. **The caller is primarily a non-frontier agent.** The surface must teach the
   workflow from the tool list alone (the way `model_status`/`model_load`/`model_unload`
   already cluster), carry the smallest possible per-call surface, and return
   machine-branchable results — not human prose. Latency budget is asymmetric: the agent
   has *endless patience* for setup but calls the measurement *in a loop*, so expense
   must live in setup and the hot path must be cheap.

3. **The position regime already proved the pattern.** ADR-001 ships as a *two-part
   instrument*: a build step (`build_axis_null.py`) that does the expensive work once and
   emits a self-describing, model-keyed artifact, plus a one-call tool
   (`analyze_axis_alignment`) that consumes it with significance already baked in. That
   is exactly the shape facts (1) and (2) demand. We extend it rather than invent.

The position regime passes its *axis* per call (two anchor strings) and prebuilds only
the *null*. The bearing regime cannot: its axis must be SVD-validated before it can be
trusted, which is expensive and belongs in setup. So the bearing artifact bundles **axis
+ validation verdict + null**, and the divergence from the sibling is deliberate.

## Decision

Ship the bearing primitive as an MCP **tool family** sharing one `bearing_analysis`
object, with verbs that read as a lifecycle:

| Tool | Role | Latency |
|---|---|---|
| `initialize_bearing_analysis` | Embed the anchor grid; **validate** the axis (union-SVD spectrum + within/inter-axis coherence + length-confound → verdict); build the measured-displacement null from precomputed real-text deltas; emit the self-describing artifact. Returns artifact ref + validity verdict. | patient |
| `run_bearing_analysis` | Project a displacement against an initialized artifact → signed component (σ), orthogonal residual (σ), cosine alignment. | fast |
| `get_bearing_analysis_status` | Inspect an artifact's verdict / provenance / embedder before running. Parallels `model_status`. | instant |

Both setup and measurement are MCP tools (not a shell script + a tool) because the
non-frontier agent drives the whole lifecycle through the tool surface, and because this
is affordable: the heavy bulk embedding is *already done* (the thought-vault corpus is
embedded), so `initialize` consumes precomputed vectors rather than re-ingesting at
scale — it is not the bulk data-plane job that ARCHITECTURE.md keeps outside MCP.

### The artifact (self-describing; the contract's load-bearing object)

A manifest + `.npy`, carrying everything `run` needs to **refuse misuse**:

- canonical `model_name` (per ADR-002) + **embedder identity** + dimensionality (e.g. 768
  for embeddinggemma vs 4096 for NV-Embed-v2)
- `axis`: the validated axis vector(s)
- `axis_validation`: SVD spectrum, coherence figures, verdict
  (`one-axis | n-axes | UNRESOLVED | flat`), face-validity record
- `null`: protocol, source-corpus identity, mean/std, count, and self-validation
  diagnostics (convergence vs subsample N, bootstrap CIs, distribution shape)
- `provenance`: anchor source + null-corpus source (enables circularity detection)

### `run` reads the embedder from the artifact, not from the agent

The agent passes only `{displacement, artifact_ref}`. The embedder is pinned by the
artifact. This resolves three constraints at once: 0001's single-embedder rule is
enforced automatically (no mismatch possible), rule #14 is honored (the model choice was
made *explicitly* at `initialize`, never silently defaulted), and the weak agent has the
minimum call surface and cannot get the embedder wrong.

### Cross-cutting rules (encoded in the contract, not left to caller discipline)

1. **Null-exemption typing.** Only embedding-derived magnitudes (`signed_component`,
   `residual`) are null-standardized; raw counts / string facts pass through
   un-normalized. Each output field declares its type. (From ADR-LNF-0002 §3; `cosine`
   is a bounded ratio — its null treatment is an open question below.)
2. **Embedder-*validated*, not just labeled.** `run` refuses if the artifact's embedder ≠
   the resolved embedder. The NV-Embed 4096-d anisotropy rationale in 0001 does **not**
   transfer to embeddinggemma 768-d for free; null and class-separation must be
   re-measured on the embedder actually in use.
3. **Circularity guard.** The artifact records build/null provenance; `run` warns (or
   refuses) when apply-data overlaps build-data. ("Gate the dataset or measure the
   window, never both on the same data.")
4. **Falsification-shaped confidence.** Significance is σ-vs-measured-null on *held-out*
   displacements; a too-clean / too-high-yield result is surfaced as an **alarm**, not a
   win. A "clean-looking curve" is not the deliverable.
5. **Target-state forms.** Designed against the post-cutover surface: per-call canonical
   `model_name` + `base_url` (ADR-003/#2, ADR-002/#11); no `model_load`/`model_unload`
   dependence; no backend-prefixed model identities.

### Errors instruct, they don't just report

Calling `run` before `initialize`, or against a `flat`/`UNRESOLVED` axis, returns e.g.
`{"error": "axis not validated (verdict: flat); call initialize_bearing_analysis first"}`.
The error names the next action, for a caller that reads tool names and error strings
literally.

### The measured null

Built from **consecutive within-conversation displacement deltas** of a real-text
corpus (real *motion*) — not random cross-conversation pairs, which measure arbitrary
jumps. The thought-vault corpus (~80,520 turns, embeddinggemma 768-d) supplies far more
samples than mean/std estimation needs (relative std error ≈ 1/√(2N): ~0.7% at
N=10,000), so the corpus size is treated as an *asset that lets the null self-validate*
— convergence vs N, bootstrap CIs, and distribution-shape checks — rather than a
sufficiency question. Non-stationarity over an evolving corpus is the real threat and is
measured, not assumed.

## Rationale

The pattern is proven (ADR-001), the layering is sanctioned (precomputed-vector consume
is not bulk ingestion), and the shape falls directly out of the two priorities: *one-step
reproducibility* forces the expensive confidence procedure into the build artifact, and
*non-frontier agent ergonomics* force a lifecycle-named family, an embedder-pinned hot
path, and instructive errors. "Spike A" stops being a standalone experiment and becomes
`initialize`'s validation phase — exactly where axis-validity belongs, as a precondition
the artifact records, not a runtime detail.

### Positive Consequences

- Axis-validity becomes a recorded precondition: `run` cannot project onto an unvalidated
  or flat axis, closing 0001's "confident projection onto a meaningless direction"
  failure at the contract level.
- The hot path is O(embed input + dot products): no per-call null rebuild, no per-call
  re-validation, no embedder negotiation. An agent can loop on it.
- Reproducibility is a property of the *artifact*, not of anyone's discipline: the manifest
  records embedder, validation verdict, null protocol, and provenance, so a result is
  re-derivable and self-describing.
- Sets the alignment target for the position regime (`initialize_axis_analysis` /
  `run_axis_analysis`) without forcing that rename now.

### Negative Consequences

- Three tools where the position regime shipped two; a marginally larger surface for a
  weak agent to navigate (mitigated by the shared `bearing_analysis` stem and lifecycle
  verbs; `get_bearing_analysis_status` is the one droppable-for-v1 piece).
- Bundling axis + null into one artifact diverges from the sibling's per-call axis,
  splitting the mental model across the two regimes until the position regime is realigned.
- The build step now owns statistical self-validation (convergence, bootstrap, shape) —
  real implementation cost, justified only by the "patient setup" budget.

## Alternatives Considered

### Option A: One MCP tool that does everything per call
**Why rejected:** It would rebuild the null and re-validate the axis on every
measurement — fatal to the fast-interaction priority — or silently cache, reintroducing
the stateful coupling ADR-003 is removing. The build/consume split is what makes the hot
path cheap and the result reproducible.

### Option B: Keep the build as a shell script (mirror `build_axis_null.py` exactly)
**Why rejected:** The primary caller is a non-frontier agent that drives MCP tools, not
shell. A script the agent can't see in its tool list breaks "teach the workflow from the
list." Affordable to promote to a tool because the bulk embedding is already done.

### Option C: Pass the embedder per call on `run` (mirror `analyze_axis_alignment`)
**Why rejected:** It invites cross-embedder mismatch (violating 0001's single-embedder
rule) and enlarges the call surface for a weak agent. Pinning the embedder in the artifact
enforces the rule structurally and honors rule #14 (explicit choice at `initialize`, never
a silent default).

## Open Questions

- [ ] **Is mean/std the right null summary?** If the projected-delta distribution is
  skewed/multimodal (plausible for an evolving corpus), σ-thresholds mislead.
  **Resolution trigger:** the `initialize` self-validation reports shape; if non-Gaussian,
  switch to an empirical-quantile null.
- [ ] **`cosine` null treatment.** It is a bounded ratio, not a magnitude — is it
  null-standardized (rule 1) or reported raw? **Resolution trigger:** decide during
  implementation against whether raw cosine is interpretable at the working dimensionality.
- [ ] **`get_bearing_analysis_status` in v1?** Kept in this proposal; it is the one tool
  droppable without breaking the lifecycle (`initialize` already returns the verdict).
- [ ] **Position-regime realignment.** When (if) `analyze_axis_alignment` is renamed to
  `run_axis_analysis` + `initialize_axis_analysis` for cross-regime naming symmetry. Out
  of scope here; a rename of a shipped tool.
- [ ] **semantic-forge schema integration.** Exact field replacing `mean_velocity` in the
  DPO/ORPO JSONL (carried from ADR-SKMCP-0001 OQ3).

## Supersession Relationships

**Supersedes:** — (additive; operationalizes ADR-SKMCP-0001, does not replace a tool)
**Superseded by:** TBD — if the escalation axis collapses to a single dominant direction
(0001 OQ1 / Spike A), `initialize` may emit a single-axis artifact and the family may
narrow; if the position and bearing regimes are unified under one "measure behavioral
axis" surface, this contract folds into it.

## Notes

The honesty marker for this ADR: the contract is **not novel architecture**. It is the
ADR-001 build-validated-artifact + one-call-consume pattern extended one regime over, with
the agent-ergonomic and statistical-discipline rules made explicit. The value is the
*discipline encoded as refusals* — validation-as-precondition, embedder-pinning,
circularity and falsification guards — not the structure, which is borrowed. Every
cross-cutting rule here was lifted from prior incoming-ideas drafts (LNF-0001/0002,
FORGE-0001, the persona ADR) that are themselves Opus-authored and self-deflating; their
humble register is itself a register and was *not* taken on faith — each rule survived
because it cashes out as a concrete refusal the tool can perform, not because the prose
was persuasive. If a rule below ever cannot be expressed as a check `run` or `initialize`
actually executes, it does not belong in this contract.
