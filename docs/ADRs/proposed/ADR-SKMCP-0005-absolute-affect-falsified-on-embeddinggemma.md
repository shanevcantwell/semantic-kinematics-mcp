# ADR-SKMCP-0005: Absolute affect/jolt measurements falsified on embeddinggemma-300m — contrastive drift survives

**Status:** proposed
**Date:** 2026-07-09 (US/Mountain)
**Author:** shanevcantwell, with Claude (Claude Code) as drafting collaborator
**Related:** ADR-SKMCP-0001 (directional-projection primitive); ADR-SKMCP-0003 (context-conditioned embedding atom / jolt — this ADR records its EG-300m result); ADR-SKMCP-0004 (affect-geometry substrate decision — nv-embed-v2, not embeddinggemma); sk-mcp issue #18 (rename `deadpan_score`/`heller_score` — mooted by this ADR); semantic-forge (sibling consumer, `README.md` JSONL example)

**Supersedes:** —
**Superseded by:** —

> **Kind:** This is a falsification record (a tombstone), not a design ADR. It closes
> a question rather than opening a build. Nothing here deletes code; it records a
> measured result and its scope.

---

## Context

Three absolute-magnitude measurements were built or proposed against
**embeddinggemma-300m (EG-300m, 768-dim)** — the substrate every existing null
artifact in `data/nulls/` was built on:

1. **`deadpan_score` and `heller_score`** — a set of ~3 absolute "affect" composite
   scores computed in `semantic_kinematics/mcp/commands/trajectory.py`
   (`compute_deadpan_score`, `compute_heller_score`), surfaced in the
   `analyze_trajectory` MCP tool output, the Gradio trajectory tab, and
   `docs/trajectory-analysis.md` / `docs/ARCHITECTURE.md`.
2. **The `jolt` concept** — absolute displacement-magnitude event detection
   (`semantic_kinematics/bearing/jolt.py`: `DisplacementNull`,
   `ConditionedDisplacementNull`, `score_jolts`), decided in ADR-SKMCP-0003.
3. *(Implicitly, the affect-direction bearing track generally, per ADR-SKMCP-0004's
   FLAT escalation result — named here for completeness; ADR-SKMCP-0004 already
   records its own falsification and this ADR does not restate it.)*

**Provenance of `deadpan_score`/`heller_score` (recorded plainly, not softened):**
these fields were confabulated by an older Opus-model drafting pass over the
operator's live objection at the time, were never operator-endorsed, and
subsequently propagated through the tool schema, UI, and docs anyway. Their status
is therefore **"was never load-bearing,"** not "a deprecated feature that used to
work." This distinction matters for how the finding below is read: it is not a
regression, it is a correction of standing.

**The `jolt` measurement**, by contrast, *was* a properly-decided instrument
(ADR-SKMCP-0003) run under a pre-registered falsification protocol. Its 2026-06-16
result (recorded in ADR-SKMCP-0003 itself) was **VERDICT: NEGATIVE** — the
conditioned-displacement-magnitude channel could not separate the deadpan
punchlines from equal-format scene-shift controls on EG-300m, at any context-ramp
`k`. That result stands as built; this ADR folds it into the same generalizable
finding as `deadpan_score`/`heller_score` because both are absolute-magnitude reads
on the same substrate.

**One measurement was not tombstoned and does not belong in this record's list of
negatives:** **contrastive drift** — differential cosine-distance between
contrastive exemplars — performed well on the one dataset run through it and
carries metadata of continuing interest. It is explicitly excluded from the
falsification below; see Decision §2.

## Decision

1. **Three absolute-magnitude measurements are falsified, scoped specifically to
   EG-300m:**
   - `deadpan_score` and `heller_score` — statistically below EG-300m's noise
     floor. Never operator-endorsed; confabulated over live objection.
   - `jolt` (absolute displacement-magnitude event detection, ADR-SKMCP-0003) —
     measured completely flat on EG-300m under the pre-registered protocol.

2. **Contrastive drift survives and is explicitly NOT tombstoned.** Differential
   cosine-distance between contrastive exemplars performed well on the one dataset
   tested and remains the prioritized surviving track. No action in this ADR
   applies to it, and no future reader should treat its adjacency to the above
   three as grounds to deprioritize it.

3. **Generalizable finding (the value banked here).** Working explanation fitting
   3 dead absolute measures + 1 surviving contrastive measure — **n=4, a working
   explanation, not a proven law:**

   > On an isotropy-regularized / cone-flattened embedder like EG-300m (see
   > ADR-SKMCP-0004: the spread-out regularizer pushes random input pairs toward
   > similar low-order statistics, homogenizing off-MTEB-objective directions),
   > **absolute-magnitude measurements sit against the embedder's own noise and
   > fall below it**, because the isotropy regularizer co-compresses signal with
   > background — there is no floor left under the signal to stand on.
   > **Contrastive / differential measurements survive** because the anisotropy
   > ("cone") the regularizer fights is common-mode across the two exemplars in a
   > contrast pair, and cancels in the differencing — common-mode rejection.

   This retroactively explains all three negatives above (they are all absolute
   reads) and prospectively predicts that contrastive measurements — e.g.
   semantic-forge's grammatical-mood `validate_diversity` — are the ones worth
   pursuing on this substrate.

4. **Scope: nv-embed-v2 is deprioritized, not tested, and not a closed negative.**
   All three falsifications above are scoped to EG-300m only. ADR-SKMCP-0004
   already names nv-embed-v2 as the correct substrate for affect/bearing work
   precisely because its cone is preserved and characterized rather than
   regularized away. A re-test of absolute measurements on nv-embed-v2 is
   **possible-but-deprioritized** — a door left ajar with low operator
   expectation — not required, and this ADR makes no claim about what such a
   re-test would find.

## Rationale

The three-part test for recording this as an ADR:

- **Hard to reverse** — undoing this record means re-litigating three measurements
  against the same noise floor; the cost of getting the scope wrong (e.g. a future
  reader assuming this falsifies nv-embed too, or assumes contrastive drift is
  suspect) is a wasted re-run or a wrongly-abandoned surviving track.
- **Surprising without context** — a future reader will find `deadpan_score`,
  `heller_score`, and `jolt` still present in source and will reasonably ask "why
  is this still here if it's dead," or worse, will build on it. The provenance
  detail (confabulated over objection, never endorsed) is exactly the kind of fact
  that is invisible from the code and must be recorded once, here.
- **Real trade-off** — the alternative to this ADR was silently deprioritizing the
  three measurements or scrubbing their references. Both were considered and
  rejected (see Consequences): a scrub destroys the marker a cold reader needs;
  silent deprioritization leaves the "was this tried?" question open forever.

## Consequences

- **Moots sk-mcp issue #18** ("Rename compat-held output schema fields:
  `deadpan_score`, `heller_score`"). Renaming a falsified field is pointless —
  there is no compat surface worth preserving under a new name for a measurement
  that was never load-bearing and sits below noise. **Action:** close issue #18
  with a pointer to this ADR.
- **`semantic_kinematics/bearing/jolt.py` and ADR-SKMCP-0003 stand as built.** This
  ADR records their EG-300m *result*, it does not delete code, revert the ADR, or
  retract the module. The instrument was correctly built and correctly falsified
  under its own pre-registered protocol; that is a clean negative, not a defect.
- **No source scrub.** The operator has explicitly declined to scrub or rename the
  propagation sites listed below — the effort is not judged worth it, and this
  tombstone is intended to do the work a scrub would: give a cold reader a marker
  at the definition site and at this record, without touching working code. The
  index below is deliberately an index, not a change list.
- **Cross-repo surface:** `semantic-forge/README.md` (sibling repo, not touched by
  this ADR) contains a JSONL example with `deadpan_score` in its DPO/ORPO schema
  illustration (`chosen_trajectory`/`rejected_trajectory` fields). That file is out
  of scope for this ADR (different repo) but is worth a pointer for whoever next
  edits that README, since the field it illustrates is now known-dead.

### Propagation-site index (informational — not a scrub list)

**`deadpan_score` / `heller_score`:**
| File | Lines |
|---|---|
| `semantic_kinematics/mcp/commands/trajectory.py` | 26–27, 121, 126, 141, 145, 268, 386, 409, 414, 431, 435, 511–512, 664, 666, 671, 673, 714–715, 720–721 |
| `semantic_kinematics/ui/tabs/trajectory/ui.py` | 99, 104, 144, 247–248 |
| `semantic_kinematics/ui/tabs/trajectory/handlers.py` | 44, 397, 401, 526, 530, 536, 540 |
| `scripts/smoke_jolt.py` | 87, 113, 125, 139 |
| `scripts/ramp_deadpan.py` | 96 |
| `docs/trajectory-analysis.md` | 5, 67, 73, 89, 94 |
| `docs/ARCHITECTURE.md` | 293–294 |
| `semantic-forge/README.md` (sibling repo, informational only) | 153–154 |

**`jolt` (module and concept references; not exhaustive line-by-line, file-level):**
| File | Note |
|---|---|
| `semantic_kinematics/bearing/jolt.py` | The instrument itself — `DisplacementNull`, `ConditionedDisplacementNull`, `score_jolts`. Stands as built. |
| `semantic_kinematics/bearing/conditioned.py` | Conditioned-atom support for the jolt track. |
| `semantic_kinematics/bearing/__init__.py` | Exports. |
| `tests/test_bearing_jolt.py`, `tests/test_conditioned_null.py` | Test coverage for the instrument (still valid — tests the mechanism, not the EG-300m verdict). |
| `scripts/smoke_jolt.py`, `scripts/spike_b_hhgg_jolt.py`, `scripts/ramp_deadpan.py` | Fixtures/harnesses that produced the falsifying runs. |
| `docs/trajectory-analysis.md`, `docs/ARCHITECTURE.md` | Doc references. |
| `docs/SPINE.md`, `docs/HANDOFF.md` | Standing project narrative — should eventually point here; not edited by this ADR. |
| `docs/research/residual-stream-jolt-survey.md` | Related research note. |
| `docs/ADRs/proposed/ADR-SKMCP-0003-context-conditioned-embedding-atom.md` | The decision + pre-registered result this ADR folds in. |

## Open Questions

- [ ] **Does the generalizable finding (isotropy regularization → absolute
  measurements fail, contrastive measurements survive) hold outside n=4?**
  **Resolution trigger:** semantic-forge's `validate_diversity` (grammatical-mood
  contrastive measurement) is the next natural test of the prediction; a
  successful or failing result there moves this from "working explanation" toward
  "law" or surfaces a counterexample.
- [ ] **Would absolute measurements (deadpan/heller/jolt) read differently on
  nv-embed-v2?** **Resolution trigger:** deprioritized per Decision §4 — only
  worth running if the nv-embed-v2 substrate work from ADR-SKMCP-0004 proceeds for
  other reasons and re-running the absolute measurements becomes near-free as a
  byproduct.
- [ ] **Issue #18 closure.** **Resolution trigger:** close with a pointer to this
  ADR (mechanical follow-up, not a research question).

## Supersession Relationships

**Supersedes:** — (does not replace ADR-SKMCP-0003's decision; folds in its
already-recorded EG-300m result as one of three negatives sharing a common cause)
**Superseded by:** TBD — a confirmed nv-embed-v2 re-test or a broader-n replication
of the isotropy/contrastive finding may narrow or extend this record.

## Notes

Calibration honesty: this is **one instrument, one explanation, four data points**
(3 dead + 1 alive), not a proven law about isotropy-regularized embedders in
general. It is recorded because it is the best current explanation that fits all
four results simultaneously and makes a falsifiable prediction (contrastive
measurements on EG-300m-class embedders will tend to survive where absolute ones
don't) — not because the sample size supports more than that. A clean negative,
banked as a gain: three measurements are now known-dead for a stated, mechanistic
reason instead of "quietly not working," and the surviving track (contrastive
drift) is explicitly protected from guilt-by-association.
