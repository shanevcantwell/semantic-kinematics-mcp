# ADR-001: Referential axis-alignment analysis (`analyze_axis_alignment`)

**Status:** Proposed
**Date:** 2026-05-29

## Context

Trajectory analysis today measures **reflexive** geometry — inter-step angles
and curvature. In 4096-D, L2-normalized embedding space those degenerate:
independently varying high-dimensional vectors sit near-orthogonal, curvature
saturates at π/2, and the signal washes out. The straight-line *displacement
ratio* (net displacement / summed path) already exists in
`compute_tautology_density` (`trajectory.py:354-364`) but is only read at its
**low tail** — `1 − ratio`, surfaced as tautology/circularity.

The new need is the mirror question with a direction attached: *does a passage
march along a **specified** semantic axis (e.g. escalation), how hard, and is
that march significant?* A **referential** projection onto an anchor-defined
axis inverts the curse of dimensionality into a precision lens — the null
projection concentrates tightly around zero, so genuine sustained alignment
flares at high sigma.

Two facts constrain the design:

- Embeddings are **L2-normalized and anisotropic** (`nv_embed_adapter.py:247`;
  cone structure). Absolute dot products are biased by the cone, so significance
  cannot be an absolute alignment — it must be a **z-score against an empirical
  null projected onto the same axis**.
- **Net displacement is interior-order-invariant** (`Σ(eᵢ₊₁−eᵢ) = e_last −
  e_first`). Shuffling sentence order does not perturb the march statistic, so a
  permutation/shuffle null is degenerate here.

There was no null-distribution, z-score, or background-corpus machinery in the
codebase — all intra-passage. This layer is net-new.

## Decision

1. **Referential, not reflexive.** Project displacement onto an anchor-defined
   axis. Do not extend angle/curvature instruments.

2. **One computation, three readouts.** Project each sentence *position* onto
   the unit axis `v̂_ref`, z-scored against the null:
   - **Position trace** `zᵢ = (eᵢ·v̂_ref − μ₀)/σ₀` — where each sentence sits on
     the axis, in sigma units.
   - **Axis drift** — the net signed march (`z_last − z_first`).
   - **Axis-restricted straightness** `|Σsᵢ| / Σ|sᵢ|`, `sᵢ = (eᵢ₊₁−eᵢ)·v̂_ref`
     — march discipline *along the axis* (the proper projected analog of the
     existing ratio). Step projections are differences of the trace.

3. **Empirical null is corpus-based, not permutation-based.** The null is the
   distribution of background-corpus embeddings projected onto *this* axis
   (`μ₀, σ₀`). This is the only null that both calibrates significance and
   removes the cone bias; shuffle nulls are excluded by the order-invariance
   above.

4. **Bring-your-own background, by reference.** The null corpus is supplied and
   cached **locally, never committed**. The repo ships at most a tiny generic
   sample for tests. This holds the personal-data boundary that defines this
   fork.

5. **Null cache keyed by model name.** Each backend has distinct geometry;
   cached background embeddings are invalid across backends. The handler refuses
   a null whose manifest `model_name` differs from the active adapter.

6. **Multi-exemplar anchors, averaged; gated on separation.** Each pole accepts
   newline-separated exemplars, averaged into a robust pole. Pole separation
   `‖e₊ − e₋‖` is reported and gated — an "axis underdetermined" error when the
   poles embed too close. When `anchor_negative` is omitted, the negative pole
   is the **null-corpus mean** (≈ cone center), so the axis points from generic
   center toward the positive concept and de-means the anisotropy in one move.

7. **Two pieces; MCP-only first cut.** (a) a **null-cache builder**
   (`scripts/build_axis_null.py`) that embeds a corpus once and persists
   embeddings + manifest; (b) the `analyze_axis_alignment` tool. No UI yet.

8. **Validation split.** Deterministic fake-adapter tests in CI (no NV-Embed).
   The real exercise — a large real-exchange corpus as the empirical null,
   absurdist books as the signal that should stand out at high sigma — is a
   **local** validation, not committed.

## Consequences

- A directional-alignment readout with **honest, cone-corrected significance** —
  the toolkit gains an active "compass" alongside the existing rhythm detectors.
- **Costs:** embedding the background corpus is a one-time heavy compute *per
  backend*; disk for cached embeddings (e.g. ~40K × 4096 × f32 ≈ 0.65 GB);
  per-call cost is a single matvec (cheap, even at 40K rows).
- **z-score meaning is tied to the chosen null** — "relative to this background
  population." Documented at the tool surface, not hidden.
- **Deferred:** Gradio UI tab; canonicalizing *which* corpus is null vs. signal;
  persistence of computed axes.

**Cross-refs:** `mcp/commands/axis_alignment.py` (implementation),
`trajectory.py:354-364` (existing ratio), `trajectory.py:185-194` (tokenize +
`embed_batch`), `embeddings/base.py:34-60` (adapter), `state_manager.py:76-112`
(adapter + cache), `server.py` (dispatch).

**Open threads:**
- ADR numbering — project-local `ADR-001` vs. the `ADR-CORE-NNN` scheme from
  `adr-namer-draft.sh`.
- Null-cache location/manifest schema as it hardens (staleness vs. model name).
- Which of the three readouts leads the tool's headline result.
