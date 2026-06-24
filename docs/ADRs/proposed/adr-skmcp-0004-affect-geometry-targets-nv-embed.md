# ADR-SKMCP-0004: Affect-direction measurement targets nv-embed-v2, not embeddinggemma-300m

**Status:** proposed
**Date:** 2026-06-24 (US/Mountain)
**Author:** shanevcantwell, with Claude (orchestrator) as drafting collaborator
**Related:** §8 target-state/spike plan (2026-06-13); ADR-SKMCP-0001 (bearing primitive); ADR-SKMCP-0002 (bearing tool contract); llauncher #155

**Supersedes:** —
**Superseded by:** —

> **Promotion note (2026-06-24):** Promoted from design-docs/incoming-ideas/ into this repo; sk-mcp is the canon home for sk-mcp ADRs.

---

## Context

Spike A returned **FLAT** for escalation on embeddinggemma-300m (768d): union-SVD top
component 12–14%, no cliff; within-axis coherence 0.10–0.17 (escalated−level deltas
barely cohere); length confound r=0.37 char / 0.19 token. Spike A itself left attribution
unresolved across three causes: embedding-null vs grid-too-weak-at-N8 vs **embedder-captive**.

A web check of the EmbeddingGemma paper (arXiv:2509.20354) supplies a mechanism for the
third. The model is trained with a **spread-out regularizer** that pushes a random pair of
inputs toward similar low-order statistics (a deliberate isotropy push), on top of geometric
embedding distillation and mean pooling. That objective allocates the space's capacity to
MTEB-rewarded *topical* structure and homogenizes off-objective directions. Affect / register /
escalation are not in MTEB. So the regularizer has every incentive to spend geometry on topic
and let affect collapse into the homogenized background — exactly the topic-dominates-affect
signature Spike A observed.

## Decision

Affect-direction / bearing measurement **does not proceed on embeddinggemma-300m**. The
instrument's validation target is **nv-embed-v2** (characterized anisotropy, no isotropy-pushing
regularizer of this kind, 4096-d). The discriminating experiment is Spike A re-run on nv-embed-v2
**plus one third embedder** with different pooling.

## Rationale

Measuring affect-geometry on a model trained to suppress the directional idiosyncrasy affect
lives in makes *every* result — FLAT included — embedder-captive and silent on whether
affect-geometry exists at all. Iterating on embeddinggemma therefore cannot falsify or confirm
the instrument; it only re-measures the regularizer. nv-embed's characterized cone additionally
lets the measured-displacement-null inherit a *known* anisotropy, instead of one co-compressed
with the signal by the same spread-out objective (signal and null flatten together, so the sigma
ratio is uninformative).

### Negative Consequences / Trade-off
- Forgoes cheap iteration on the already-embedded 80,520-turn embeddinggemma corpus.
- Pays nv-embed serving cost: llauncher needs a vLLM/SentenceTransformers server type (#155)
  before sk-mcp can route it; plus a full re-embed at 4096-d.

## Open Questions

- [ ] Is the FLAT **embedder-captive** (spread-out mechanism) or is affect **weakly-linear in any
  sentence embedding**? **Resolution:** cross-embedder Spike A re-run (nv-embed-v2 + one third
  embedder). Coherence jumps where spread-out isn't suppressing → embedder-captive → the
  affect-gate reopens. Stays flat across embedders with different training/pooling → wrong atom →
  pivot to the residual-stream channel (StALT, §8 rev 2026-06-15(c)), not a better embedder.

## Supersession Relationships

**Supersedes:** partially supersedes the §8 (2026-06-13) standing decision to run **Spike A→B
falsify-fast on current tooling** — that sequencing assumed current-embedder iteration was
informative; this ADR holds it is not, for the affect axis specifically. (Update §8's note to
reference this ADR.)
**Superseded by:** TBD (the cross-embedder re-run may convert this from `proposed` to `accepted`,
or overturn the premise and retire it).

## Three-part test (why this is an ADR)

- **Hard to reverse** — substrate commitment for the whole bearing track + re-embed cost.
- **Surprising without context** — a future reader asks why the model already embedded at 80k
  turns wasn't just pushed harder.
- **Real trade-off** — cheap iteration on owned infra vs measurement that can't be informative.
