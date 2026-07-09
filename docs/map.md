# The instrument map — the wider target constellation

Companion to [`SPINE.md`](./SPINE.md). The SPINE holds the **three founding aims**
(segmentation, behavioral tracking, longitudinal generational drift), the absurdist
MacGuffin, and the semantic-forge downstream. This file holds the **wider constellation** —
newer targets for the *same one instrument* that are not among the founding three.

**This is a map, not a roadmap.** Nodes are visitable in any order; none blocks another; the
whole set rarely needs to be in view at once. Every node sits on one axis — the selection
filter the whole program prunes by:

> **cheap compute · mathematical (not semantic) · contrastive (not absolute) · falsifiable.**

That filter is why these have a chance at *legitimacy*, and why the absolute/semantic
measurements (`deadpan_score`/`heller_score`/jolt-magnitude) were falsified — see
[ADR-SKMCP-0005](./ADRs/proposed/ADR-SKMCP-0005-absolute-affect-falsified-on-embeddinggemma.md).
A mathematical measure has a null, a noise floor, and a claim that can be *wrong in a
checkable way*. A semantic (LLM-judge) measure costs a forward pass, doesn't reproduce, and
inherits the judge's completion/escalation bias — the exact mechanism that confabulated the
falsified scores.

---

## Node — Intelligent `assert()`  ·  consumer: prompt-prix
- **Measurement:** a *contrastive* assert — `drift(actual, reference) < band` between a
  tool-call/completion and a golden exemplar, same embedder. NOT absolute region-membership.
- **Promise:** a semantic test oracle for tool-usage tests — catches "right intent, different
  phrasing" that a literal `assert()` false-fails, with no model-in-the-loop judge.
- **Constraint:** embedding proximity ≠ functional equivalence for *discrete* args
  (`user_id=5` vs `6` is a microscopic step but a decisive error). Fits the NL/intent layer;
  keep exact-match for discrete args. Choosing the wrong assert type for the layer is the
  failure mode.
- **Free calibration:** tool-usage tests carry ground truth → embed known-good vs known-bad,
  set the band where they actually separate. prompt-prix is generator *and* consumer.
- **Cheap-first test:** on one labeled test set, check that contrastive drift separates
  pass/fail *above noise* before wiring it in.
- **Relations:** survivor class (contrastive); same shape as semantic-forge
  `validate_diversity`; inherits issue #1's null-as-session-state.
- **Status:** unexplored.

### sub-node — Seeking / thrash detection
- **Measurement:** a trajectory *shape*, not a learned "seeking axis" — sustained low
  inter-call drift + low net-to-gross displacement ratio (tortuosity / straightness).
  Seeking = motion without progress, circling not advancing. The signal is regime entry/exit
  (boundary detection — the instrument's native framing).
- **Promise:** cheap agent-monitor flag for "stuck in a seek loop" (aim-2 / behavioral),
  generalizes past filetrees to any repeated-tool-with-tweaked-args thrash.
- **Constraint:** procedural-first — *lexical* seeking (`/util` vs `/utils`) is caught by
  string metrics (edit distance, shared-prefix); embeddings earn their place only for
  *semantic* seeking (`/auth`→`/login`→`/session`). Short N (3–8) → tortuosity is noisy:
  "this run is seeking" is reliable, "seeking started at call 3" is not.
- **Cheap-first test:** on the existing filetree specimen, race string-metric straightness vs
  embedding-drift straightness — if strings already separate it, embeddings are unnecessary
  here (banked negative).
- **Status:** unexplored; one specimen in hand.

## Node — Longitudinal per-user abrupt-change  ·  origin: TVI (co-dreamed)
- **Measurement:** boundary/event detection on a *per-user* longitudinal embedding trajectory
  (across sessions, not within-document). Each user measured against their own history.
- **Promise:** the cleanest common-mode-rejection setup on the board — self-baseline means
  the instrument cone *and* the user's voice are both common-mode within their own stream and
  cancel; the cone-null we agonize over elsewhere is *free* here. Population scale (100Ms) is
  the null that calibrates what "abrupt" means.
- **Constraint — the hard part is NOT the detection:** (a) *attribution* — abrupt change ≠
  crisis; a boundary has a dozen benign causes; calling it intervention-worthy is re-detecting
  *change itself* and naming it meaning (Sylos-Labini, but the ghosts are people). (b) *base
  rate* — 0.1% false-positive × 100M = 100k wrongly flagged; the absolute FP count, not the
  rate, is the design target. (c) intervention *policy* is where all benefit/harm lives;
  consent/governance gate who can even hold 100M streams.
- **Cheap-first test:** thought-vault *is* longitudinal — prove the boundary detector on the
  operator's own single stream (self-baseline, no consent question) before the scaled version
  is relevant.
- **Relations:** same instrument, new timescale. New *longitudinal-atom* question
  (user-rep-at-time-t = session-summary embedding? rolling window?) — ADR-SKMCP-0003's
  conditioning problem one level up.
- **Status:** unexplored; single-stream POC available on the tvi corpus.

---

## Datasets & provenance (in play)

**nv4096 — NV-Embed-v2 (4096-d) embedded dataset.**
`thought-vault-integration/output/thought-vault-integration-data/nv4096/corpus_4096.jsonl`.
Produced by the TVI pipeline; `meta.json` = `NVEmbed:nvidia/NV-Embed-v2`, 4096-d.

- **First analysis touch: 2026-07-09** (cone characterization) — this *updates* SPINE's
  "untouched by analysis to date" / "the embeddings have not been touched by analysis."
- **Measured cone (voice-loaded — instrument + operator voice, inseparable from one corpus):**
  ‖μ‖ = 0.554 (~80× the isotropic baseline 1/√N), mean pairwise cosine = 0.308, participation
  ratio ≈ 34 effective dims (top-50 dims = 54.5% of variance). Strongly anisotropic,
  moderate-rank. Non-zero rows are exactly unit-normalized (matches sk-mcp
  `nv_embed_adapter` F.normalize).
- **⚠ Pipeline defect:** 11.74% of sampled rows are EXACT zero vectors — a systematic
  embed-failure population (~11.5k chunks extrapolated), not noise. Must be filtered from any
  null/cone estimate; a bug against the TVI pipeline is warranted.
- **⚠ Figure discrepancy:** the actual file is **98,293 lines**; SPINE records **85,570 /
  87,004** (tvi#43). Needs reconcile.

---

*Placement: this companion keeps the read-first SPINE focused on the founding three. If a
node here graduates to a founding aim it moves into SPINE (one home per concept,
ADR-CON-0001). The constellation is cross-repo by nature — it may distil to the design-docs
pools if it outgrows this repo.*
