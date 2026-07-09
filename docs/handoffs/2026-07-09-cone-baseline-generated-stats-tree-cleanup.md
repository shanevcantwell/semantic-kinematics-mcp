# Handoff — 2026-07-09 · cone baseline, generated-stats, tree cleanup

**For the cold next session.** Durable framing is [`docs/SPINE.md`](../SPINE.md); this file is
the session-continuity note. (The older `docs/HANDOFF.md` predates 2026-07-09 and carries
retired figures — read SPINE + this + the generated report for current state.)

## TL;DR — what happened

- **First analysis touch of the nv-embed 4096-d corpus** (it was "untouched" until today):
  cone characterized — ‖μ‖=0.554 (voice-loaded), mean pairwise cos 0.308, participation ratio
  ≈34 effective dims. Strongly anisotropic, moderate-rank.
- **A clean NEGATIVE banked** — `ADR-SKMCP-0005`: deadpan/heller/jolt-magnitude falsified on
  embeddinggemma-300m; the generalizable finding is *absolute measures die below noise on a
  cone-flattened embedder, contrastive/differential ones survive*. `deadpan_score`/`heller_score`
  were never operator-endorsed (old-Opus confabulation); #18 closed as mooted.
- **Generated-stats adopted** — `ADR-SKMCP-0006` + `scripts/summarize_corpus.py` →
  `docs/generated/CORPUS_STATS.md`. Derived facts are generated, never hand-typed; SPINE/map
  now point at the report. This killed a whole class of staleness (see "conflation" below).
- **The wider target map** — `docs/map.md`: intelligent-`assert()` (prompt-prix), seeking/thrash
  detection, longitudinal per-user drift (TVI-origin), all on the *cheap·mathematical·contrastive·
  falsifiable* filter.
- **Non-frontier work-package** — `docs/plans/nv4096-contrast-panel-workpackage.md`: reuse-not-build
  instructions to embed a Nemotron contrast corpus through the byte-identical nv-embed path.
- **Working ground reaped clean** — 6 squash-merge leftover branches + the entire `fix/34-*` knot
  removed (all residue; #34 was already fixed on main via #39; PR #38 closed superseded; #48 closed
  as a false-alarm — the "stranded" commit's content was already on main).

## Current true state (from `docs/generated/CORPUS_STATS.md`, regenerate to confirm)

- **768-d embgemma store: 100%** (87,004/87,004). — this is the store prior sessions kept
  reporting as "done."
- **4096-d nv-embed store: 99.71%** (86,748/87,004, **256 OOM-stuck** — tvi#43). This is the
  research substrate (ADR-SKMCP-0004). NOT cone-nulled or validated against the aims yet.
- The two were being **conflated**; that is now structurally impossible (one derived table,
  model+dim explicit).

## What's next — OFFLINE / operator-gated (no single prior home; enumerated here)

| Step | Home | Operator action |
|------|------|-----------------|
| `/mnt/storage` migration (data-lake + per-dim self-describing layout) | **tvi#45** | 3 decisions: does `chunks.jsonl` move? · 768 model-id · go to touch live writer `orchestrator.py` |
| 768-d store identity (mint its missing `.meta.json`) | **tvi#37** | supply the canonical model-id string (mint-once) |
| 256 OOM residue disposition | **tvi#43** | sub-chunk / truncate-tag / exclude (title still says stale "1,434" → is 256) |
| Corpus backup off working tree | **tvi#38** | rclone → GDrive; = the coming `/mnt/storage` backup strategy |
| Nemotron contrast-panel embedding run | **work-package** | GPU on RTX-8000; batch run (yours or a non-frontier orchestrator) |

**Recommended tvi ADR follow-up:** the storage-architecture decision (`/mnt/storage` as data-lake,
git repos hold code + generated reports, per-dimension self-describing layout) is ADR-worthy but
was left as tvi#45 pending the 3 decisions above — promote to a tvi ADR when they resolve.

## Pointers

- Framing: `docs/SPINE.md` · Targets: `docs/map.md` · Live counts: `docs/generated/CORPUS_STATS.md`
- Decisions: `ADR-SKMCP-0005` (falsification), `ADR-SKMCP-0006` (generated stats)
- Plan: `docs/plans/nv4096-contrast-panel-workpackage.md`
- Cross-repo: tvi#45 (migration), tvi#37 (768 identity), tvi#43 (256 residual), tvi#38 (backup)
