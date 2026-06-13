# ADR-SKMCP-0001: Directional Projection Primitive for sk-mcp

**Status:** proposed
**Date:** 2026-06-02 (US/Mountain)
**Author:** shanevcantwell, with Claude (Opus) as drafting collaborator
**Related:** semantic-kinematics-mcp (the tool surface being extended); semantic-forge (downstream consumer — JSONL schema, CogSec judge calibration); the three-axis escalation hypothesis (the first behavioral axis to be measured)

**Supersedes:** —
**Superseded by:** —

> **Promotion note (2026-06-13):** Promoted from `design-docs/incoming-ideas/`
> into this repo; sk-mcp is the canon home for sk-mcp ADRs, and the source copy in
> `incoming-ideas/` is removed so no duplicate diverges. The `SKMCP-NNNN` identity
> is retained deliberately — cross-repo references (semantic-forge, design-docs)
> depend on it — rather than renumbering to a local `ADR-004`; the
> `ADR-00N` vs `ADR-SKMCP-NNNN` scheme reconciliation remains the open ADR-numbering
> thread carried from ADR-002/003. **Relationship to ADR-001:** ADR-001 (referential
> axis-alignment) is the *position-regime* sibling — sentence atom, corpus null;
> this ADR is the *bearing-regime* primitive — displacement atom, measured-displacement
> null. Additive, not superseding. See `docs/HANDOFF.md` §8 for the regime split.

---

## Context

sk-mcp measures **magnitude** richly and **bearing** not at all. Every metric in the current surface collapses direction:

- `analyze_trajectory` velocity is the L2 norm of sentence-to-sentence displacement — the README's own Known Limitation states it discards direction. Acceleration compounds this.
- The two angle-shaped measures it *does* expose are both the **reflexive** kind: `calculate_drift` is the angle between two data points (cosine), and curvature is the angle-change between consecutive displacement vectors. In ~4096-d (NV-Embed-v2) both saturate toward orthogonality — the trajectory-analysis design note confirms curvature clusters near π/2 for all text and discriminates nothing.

What is absent is the angle of a motion relative to a **named reference direction** — a projection. This became load-bearing the moment a behavioral axis (escalation: tone/urgency/importance) needed to be measured, because escalation is a *directional* claim about text motion, and the toolkit can only say how far a completion moved, never which way. The omission is a fossil of what the trajectory layer was built for (absurdist/TAT rhythm signatures, which are magnitude-and-cadence phenomena where bearing toward a named axis is irrelevant). The tool is correctly specialized for the rhythm regime; escalation is the mirror regime and needs the primitive that regime never required.

## Decision

Add a **directional projection tool** to the sk-mcp MCP surface.

1. **Inputs:** a displacement (either the delta between two texts, or a passage's net displacement) and a **reference axis** defined by two anchor texts whose difference *is* the axis (e.g. level-phrasing → escalated-phrasing for the tone axis).
2. **Outputs:** the **signed component** along the axis, the **orthogonal residual**, and the **cosine alignment**.
   - The signed component is the quantity escalation measurement actually wants — "how much of this motion was up the escalation axis" — distinct from velocity, which threw the direction away.
   - The residual is the other half of the diagnostic: a completion moving fast but mostly orthogonal to the axis is doing something else, and magnitude-only scoring would mislabel it high-velocity-toward-nothing.
3. **Normalization is part of the contract, not a caller's afterthought.** The signed component is reported alongside a standardization against a **measured null** of random displacements projected onto the same axis. The null MUST be measured, not assumed at 1/√d, because the embedding space is anisotropic (decoder-LLM embedders like NV-Embed-v2 have known cone structure), so baseline alignment is itself directional.
4. **Single-embedder constraint.** Projection and the axis-defining anchors must be embedded in the **same** backend within a single measurement. Distances and directions are not comparable across embedders (the embeddinggemma↔NV-Embed-v2 first_plural/passive order flip is the standing evidence). Embedder choice is part of the hypothesis, not a fixed backdrop.

## Rationale

High dimensionality *defeats* the reflexive angle measures and *sharpens* the referential one — the same fact cuts both ways:

- Curvature compares two data-derived displacements, both forced near-orthogonal, so it saturates and discriminates nothing.
- Projection compares one displacement against a fixed, deliberately-constructed axis. A random displacement's alignment with any fixed direction concentrates near zero with small spread, so a genuine axis-aligned motion stands out at high sigma rather than washing out. The dimensionality that kills curvature is what makes a bearing measurement crisp.

### Positive Consequences

- Supplies the missing instrument that the escalation hypothesis, the mood-axis cross-program test (does "yelling" direction-from-imperative align with the tone axis?), and the union-SVD collinearity question all assume exists.
- The displacement-ratio component already inside `tautology_density` (net displacement ÷ path length) is a direction-**coherence** scalar that survives the curvature problem and pairs naturally as a cheap pre-filter: ratio screens "is there sustained drift at all" on long passages; projection supplies the bearing the ratio cannot. Reuses an existing signal rather than duplicating it.
- Unblocks the semantic-forge judge calibration: the projection onto the escalation axis is what replaces `mean_velocity` in the JSONL schema, after which CogSec `manipulation_score`-vs-projection becomes the real calibration test.

### Negative Consequences

- The signed component cannot be read raw — in 4096-d it is small in absolute terms even when real — so the measured-null standardization is mandatory overhead on every call, not optional.
- Measuring the null is itself a cost (a batch of random-displacement projections per axis), and the null is axis-specific and embedder-specific, so it cannot be computed once globally.
- Axis quality is entirely dependent on anchor-pair quality; a poorly chosen anchor pair yields a confident projection onto a meaningless direction (the documented "no face validity" failure of word-embedding semantic directions).

## Alternatives Considered

### Option A: Reuse `calculate_drift` / curvature as the direction measure

**Why rejected:** Both are reflexive angles (point-to-point, or step-to-step) and saturate near π/2 in high-d. Curvature was empirically shown non-discriminating in the trajectory-analysis work. Neither can express bearing toward a *named external* axis — they only relate data vectors to each other. This is the core reason the primitive must be new rather than assembled from existing tools.

### Option B: Representation-engineering projection on the generating model's activations

The prior-art review (2026-05-29) found projection-onto-a-concept-direction is textbook RepE: anchor-pair → delta → PCA → project; the dot product is the score. RepE operates white-box on the *generating model's* residual stream, usually to **steer**.

**Why rejected (as the primitive for sk-mcp):** sk-mcp's purpose is read-only **measurement** of a behavioral property of *text* (a completion, a trajectory) via a separate **black-box** embedder, with no access to and no intent to modify the generating model. Same math, different substrate and different purpose. The RepE framing is the right intellectual lineage to cite but the wrong access model for this tool — and recording that distinction is the point, because it is the only thing that differentiates this from prior art.

### Option C: Report only the signed component (no residual, no null)

**Why rejected:** Without the residual, a high-magnitude motion orthogonal to the axis is indistinguishable from a real axis-aligned motion of the same projected length. Without the measured null, the raw component is uninterpretable in high-d and anisotropic space. Both are required for the number to mean anything; dropping them produces confident noise.

## Open Questions

- [ ] **Dimensionality of the target axis is itself unmeasured.** The escalation hypothesis is three axes (tone/urgency/importance) held as an *anchor to disprove*. Whether they are one collapsed manifold or genuinely separate is a full-vector SVD that must run before projection has a meaningful axis to project onto. **Resolution trigger:** run the union SVD (escalation grid + mood variants, same embedder) and read the singular values — coarse structure (how many dominant directions) is embedder-stable and trustworthy; marginal components (is the third axis real or the second's noise) are exactly where an embedder swap flips the answer, so treat a "two strong + one marginal" result as unresolved, not as a finding.
- [ ] **Null-measurement protocol.** How many random displacements, drawn from what distribution (uniform on sphere vs. resampled real-text deltas), constitute an adequate per-axis null under anisotropy? **Resolution trigger:** specify during implementation; resampled real-text deltas are the likelier-correct null because they inherit the space's actual anisotropy rather than assuming isotropy.
- [ ] **Schema integration with semantic-forge.** Exact field name and shape replacing `mean_velocity` in the DPO/ORPO JSONL. **Resolution trigger:** decide when wiring the judge-calibration correlation, downstream of the SVD producing the axis.

## Supersession Relationships

**Supersedes:** — (additive to the sk-mcp surface; does not replace an existing tool)
**Superseded by:** TBD — if the escalation SVD collapses the three axes to one dominant direction, a future revision may narrow this from a general axis-projection tool to a single escalation-scalar, or fold the null-measurement into a higher-level "measure behavioral axis" tool.

## Notes

The intellectual honesty marker for this ADR: the primitive is **not novel** as an operation (it is standard representation engineering, and projecting words onto a bipolar semantic direction predates RepE in the word-embedding literature). The differentiation is the application — read-only black-box-embedding measurement of a behavioral axis on the output side, versus white-box activation steering. The cautions that feel like hard-won method (no universal threshold, scores relative and noisy, poor cross-domain generalization) are documented limitations of the technique, not discoveries. Recording this keeps the tool's claim accurate: it imports a known operation into a substrate where it was not previously applied, and the value is the import plus the measured-null/anisotropy discipline, not the math.
