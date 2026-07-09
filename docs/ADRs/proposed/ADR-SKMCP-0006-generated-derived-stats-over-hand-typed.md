# ADR-SKMCP-0006: Derived stats are generated, never hand-typed — enforcement surface over prose

**Status:** accepted
**Date:** 2026-07-09 (US/Mountain)
**Related:** ADR-SKMCP-0005 (shares the enforcement-surface principle — a generated null/report over an asserted one); ADR-CON-0005 (confirm-complete bell — the (f) enforcement-surface open question below)

---

## Context

This session hit a cascade of false/stale figures in the durable record, each of which
cost real forensic time to unwind:

- the 768-d-vs-4096-d **"100% complete" conflation** (repeated across sessions),
- a mischaracterized **"~12% zero-vector pipeline defect / ~11.5k chunks lost / file a bug"**
  (actually `_failed`-tagged retry-superseded markers; true completion 99.71%, 256 stuck),
- stale **85,570 / 1,434** completion counts in SPINE/HANDOFF that a later retry pass had
  already superseded.

Every one was a **derived statistic hand-transcribed into prose** (SPINE.md, map.md,
HANDOFF.md), which drifted the moment the ever-growing embedding checkpoints changed. The
ecosystem already carried the fix pattern — llauncher's `summarize_tests.py` (ground-truth
derived, committed, "do not hand-edit") — and sk-mcp already had the compute half
(`embed_status.compute_status`, which streams a checkpoint and derives done/failed/pending/
total with correct dedup semantics).

## Decision

1. **Derived facts about a corpus are generated, not authored.** `scripts/summarize_corpus.py`
   (reusing `embed_status.compute_status`) emits `docs/generated/CORPUS_STATS.md` — per-store
   model/dim (from the `.meta.json` sidecar), done/failed/pending/total, completion %,
   raw-vs-distinct line delta, checkpoint mtime/size.
2. **The generated report is a committed, overwrite-in-place markdown artifact** carrying a
   "**Auto-generated … do not hand-edit**" banner. It is diffable and greppable (git-backed);
   the raw checkpoints stay gitignored. It is a *snapshot*, self-labelled as such — regenerated
   on demand, not a source of truth in itself.
3. **Living docs point at it; they do not carry the figures.** SPINE.md and map.md reference
   `docs/generated/CORPUS_STATS.md` for completion and retain only *non-derived* narrative
   (research findings like the cone measurement, framing). Hand-typed completion figures are
   retired.
4. **General principle (this is the reusable part):** a derived fact stated in prose has **no
   enforcement surface** and drifts; a generated artifact re-derives from ground truth and
   cannot lie. Prefer the generated artifact wherever the fact is computable.

## Rationale

### Positive Consequences
- **Staleness-immune by construction** — regenerate and it is true; there is no hand-typed
  number left to rot.
- **The 768/4096 conflation is structurally impossible** — both stores appear in one derived
  table with model + dimension explicit; no reader can attach one store's "100%" to the other.
- Establishes the `docs/generated/` convention in sk-mcp (matches llauncher, semantic-forge).
- The report doubles as the **pre-/post-migration verifier** for the `/mnt/storage` move
  (tvi#45): regenerate after any move to confirm no store forked or path dangled.

### Negative Consequences
- Regen streams the 8 GB nv-embed checkpoint (~2 min) — deliberate, not per-commit.
- The committed snapshot can lag ground truth *between* regenerations; mitigated by the
  self-labelling banner and the low cost of regenerating.

## Alternatives Considered

### Option A: Keep hand-maintained figures in prose (status quo)
**Rejected** — it is the exact failure this session diagnosed. No enforcement surface; every
figure is one checkpoint-growth away from false.

### Option B: On-demand stdout only (the `embed_status.py` shape), no committed doc
**Rejected for the record layer** — not diffable, greppable, or visible to a cold reader
without running it. Good as the *compute* primitive (and reused as such), insufficient as the
*record*.

## Open Questions

- [ ] Should confirm-complete's **(f)** condition ring from *re-running the generator* where one
  exists, instead of a bare attestation? **Resolution:** raised as an ADR-CON-0005 open question
  (the skill-edit suggestion surfaced during this session's completion ring).
- [ ] Is an incremental/cached regen mode worth it for the 8 GB store? **Resolution:** only if
  regen frequency warrants; defer until it bites.

## Supersession Relationships

**Supersedes:** — (no prior ADR; this retires *hand-typed* provenance in SPINE/map, which were
never decision records).
**Superseded by:** TBD.

## Implementation Notes

| File | Change | Ref |
|------|--------|-----|
| `scripts/summarize_corpus.py` | Created — generator (reuses `embed_status.compute_status`) | `09f2a4d` |
| `tests/test_summarize_corpus.py` | Created — 4 tests (dedup/`_failed`/sidecar/size) | `09f2a4d` |
| `docs/generated/CORPUS_STATS.md` | Created — baseline report; establishes `docs/generated/` | `09f2a4d` |
| `docs/SPINE.md`, `docs/map.md` | Retired hand-typed completion figures → pointer; corrected zero-vector provenance | `f5b3b06` |
