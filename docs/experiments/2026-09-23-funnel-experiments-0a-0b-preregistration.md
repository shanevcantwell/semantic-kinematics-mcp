# Draft protocol / preregistration scaffold: Funnel experiments 0A and 0B

**Program handle:** `SKM-FUNNEL-EXP0`
**Experiment handles:** `SKM-FUNNEL-EXP0A`, `SKM-FUNNEL-EXP0B`
**Status:** draft protocol / preregistration scaffold; empirical run status is **not
registered**, and no 0A/0B run has been performed
**Scaffold recorded:** 2026-09-23 (US/Mountain)
**Governing decision:** [`ADR-SKM-009`](../ADRs/proposed/adr-skm-009-funnel-measurement-profile.md)
**Lineage:** [`ADR-SKM-008`](../ADRs/proposed/adr-skm-008-functional-direction-probe-generalized-axis-source.md), refined but not superseded by ADR-SKM-009

This file is not a completed preregistration. A separately versioned run manifest must freeze
all choices named below before an empirical run can acquire `preregistered` or `registered`
status. In particular, the candidate, generating model, prompt population, axes, sample size,
statistics, cutoffs, and exact decision rules remain unset.

---

## Purpose and evidence boundary

This document records the hypothesis structure and draft protocol for the first two empirical
tests of the funnel-measurement profile. The H0 statements and predicted observable behavior
in this scaffold were recorded **before the experiment-layer implementation or its unit-test
behavior was observed**. That timing is preserved, but it does not freeze the many unset run
choices or turn this scaffold into a completed preregistration. The two questions are:

- `SKM-FUNNEL-EXP0A`: does a known compact directive produce a reproducible,
  target-directed population displacement that survives paraphrase controls?
- `SKM-FUNNEL-EXP0B`: when a funnel has multiple clauses, do independently measured
  constituent displacement fields align, and does their composition depart positively
  from the additive prediction along the future run-manifest target direction?

The experiment harness, embedding functions, and geometric operations may have unit tests.
Those tests establish only that implementation behavior matches its code contract. **Unit-test
success is not an empirical result for 0A or 0B and cannot change `Result` from `PENDING`.**
Only matched model outputs generated under a future, versioned, fully frozen run manifest can
count as experimental evidence. Implementation and unit-test observation after the hypotheses
were written does not alter H0 or the predicted behavior recorded here.

Null outcomes will be informative once a run is registered. A null 0A result would say that
the proposed measurable funnel object was not established under the run-manifest-frozen model,
prompt population, and instrument. A null 0B result distinguishes directional efficacy from constituent reinforcement: it can
leave an independently established H1 intact while rejecting or withholding H2/H3.

---

## Required run-manifest freeze and calibration boundary

The following are requirements for a future run, not choices frozen by this scaffold. Before
generating or inspecting any confirmatory funnel-conditioned answer, the experiment owner
must freeze them in a versioned run manifest:

1. generating model/checkpoint and invocation parameters;
2. prompt population, task-family/topic strata, and development/holdout split;
3. run/seed schedule and pairing key;
4. candidate text, constituent boundaries, paraphrases, neutral controls, and bleached
   controls;
5. embedding instrument(s), preprocessing, centering, calibration artifact,
   empirical background null, and geometry variant(s);
6. target axis and its independently labeled positive/negative examples;
7. collateral dimensions and direct behavioral labels;
8. analysis code revision, artifact schema version, and exclusion rules;
9. exact statistics, sample size, cutoffs, multiplicity/error-control policy, and decision
   rules.

No theoretical random-vector null may silently replace the empirical null frozen by that
manifest. Any centered/calibrated geometry alternatives selected for the run must be named in
advance and run in parallel. Disagreement is reported; no geometry is selected retrospectively
because it gave the preferred result.

### Numerical thresholds, sample size, and any calibration pilot

ADR-SKM-009 does not supply universal numerical cutoffs. Therefore this scaffold does not
invent or register an effect-size, cosine, confidence, sample-count, quantile, error-control,
or superadditivity threshold. Without those values, 0A/0B is not ready to run.

If null-only calibration is needed to choose any of them, it must be a **separately
preregistered pilot**, created before pilot data are generated or inspected. That pilot
registration must state exact stopping rules, sample-count bounds, quantile estimators,
stability criteria, exclusions, multiplicity/error-control rules, and the deterministic rule
that maps pilot results into confirmatory sample sizes and cutoffs. This scaffold supplies none
of those values.

Pilot prompts, seeds, and control outputs must be disjoint from the independent confirmatory
controls and confirmatory prompt/seed cells. Original-funnel, paraphrase, constituent, and
combination outputs are unavailable to pilot calibration. After the pilot, its immutable
artifacts and rule-derived values must be referenced in the versioned 0A/0B run manifest;
that manifest must freeze every item above before confirmatory generation begins. If pilot
stopping or stability requirements fail, the planned experiment is **underpowered / not
evaluable**, not negative and not eligible for post-hoc repair. A revised pilot or run requires
a newly versioned registration.

If no pilot is used, the run manifest must instead supply externally justified exact values
before any confirmatory output is generated. Until one of these paths is completed, no
empirical run status or numerical decision procedure is registered.

### Shared pairing and provenance

Every future experimental condition must use the same prompt/seed cells wherever the
generating API permits deterministic seeds. A missing or failed member invalidates that paired
cell for the comparison that needs it and remains visible in the attrition report. No answer
may be silently substituted.

Every future raw record must include at least the ADR-SKM-009 minimal contract:

```json
{
  "experiment_id": "SKM-FUNNEL-EXP0A|SKM-FUNNEL-EXP0B",
  "model": "...",
  "instrument": "...",
  "prompt_id": "...",
  "seed": "...",
  "condition": "baseline|funnel|control|constituent|combination",
  "funnel_id": "...",
  "constituents": ["F1", "F2"],
  "turn": 0,
  "response_ref": "...",
  "embedding_ref": "...",
  "provenance": {}
}
```

The future run manifest must require each frozen record to extend `provenance` with
model/checkpoint identity, invocation parameters, prompt-population version, condition-text
hash, target-axis artifact reference, calibration and empirical-null references, geometry
variant, code revision, timestamps, and exclusion reason where applicable. Derived artifacts
must be reproducible from raw records.

---

## `SKM-FUNNEL-EXP0A` — Known funnel

### Hypothesis position

This experiment tests ADR-SKM-009's H0 against H1 only.

- **H0 — No funnel:** observed original/paraphrase differences are ordinary paired run
  variation, neutral-instruction sensitivity, length effects, or a surface-form-specific
  “magic string”; they do not form a reproducible target-directed population displacement.
- **H1 — Directional funnel:** the original intervention and independently accepted semantic
  paraphrases produce a reproducible, target-directed displacement relative to matched
  no-funnel and control conditions.

0A does not test constituent alignment, superadditivity, attractor behavior, or recursive
reinforcement. It cannot support H2–H5.

### Candidate and conditions

Select one compact intervention with repeated historical qualitative evidence. The example
class from ADR-SKM-009 is:

> Respect the precise definitions when using terms of art.

The exact selected directive and evidence pointer must be frozen before output generation.
Construct these conditions:

- no-funnel baseline;
- original funnel;
- a manifest-frozen set of semantic paraphrases screened for semantic equivalence
  independently of model output (exact count currently unset);
- token/length-matched neutral instructions;
- semantically bleached controls where feasible.

The same fixed prompt population is run across the same seed schedule under every condition.
Prompt coverage is stratified by the run-manifest-frozen task families, topics, and
conversation positions; stratum identities remain in the evidence rather than being pooled
away.

### Predicted observable behavior

If H1 holds under the future run-manifest-frozen instrument:

1. paired original-funnel displacements separate from no-funnel/neutral/bleached empirical
   controls along the run-manifest-frozen target axis;
2. displacement vectors show a reproducible common direction, not only increased global
   norm;
3. the original and the run-manifest-frozen paraphrase quorum agree in direction and retain
   target projection across prompt/seed strata;
4. off-axis/residual movement and collateral dimensions remain visible and are not allowed to
   masquerade as efficacy;
5. the result is not carried by a few extreme prompt/seed cells.

A larger overall embedding shift without target-directed, population-consistent displacement
is not the predicted funnel effect.

### Planned falsifier and decision-rule shape (not yet precommitted)

The following verdict shape records the intended falsifiers, not a currently executable
decision rule. A future run manifest must freeze the exact numerical cutoffs, statistics,
stratum requirements, and paraphrase quorum through the calibration boundary above before
funnel-conditioned output is observed. Only then may the frozen analysis yield one of three
verdicts:

- **Advance H1 for independent replication** only if the original funnel and the entire
  run-manifest-frozen paraphrase quorum satisfy all of the following relative to both
  no-funnel and run-manifest-frozen instruction controls: empirical separability,
  target-directed projection, displacement-direction consistency, and population consistency
  across the run-manifest-frozen strata. The axis-free/residual report must not show that the
  apparent result is merely a larger global shift.
- **Withhold/reject H1 under the future run manifest** if the original does not separate from
  controls, the effect is not target-directed or coherent, only the exact original wording
  survives, the result depends on a few extreme cells, or collateral/global displacement
  explains the apparent gain.
- **Not evaluable** if calibration, pairing, attrition, or sample stability fails the frozen
  validity requirements. “Not evaluable” cannot be converted to support by relaxing a cutoff
  after unblinding.

A result that advances H1 remains instrument- and checkpoint-bounded until independent
embedding-instrument and direct-label replication. It is not evidence for an internal model
vector.

### Controlled method

1. Complete the separate pilot registration if calibration is needed; otherwise justify the
   exact externally specified values.
2. Freeze all shared materials, exact candidate/condition texts, and decision parameters in
   the versioned run manifest.
3. Generate matched baseline, original, paraphrase, neutral, and bleached-control outputs for
   every prompt/seed cell without adaptive regeneration.
4. Persist outputs before embedding; embed every retained output with the same
   run-manifest-frozen instrument and preprocessing.
5. Calculate paired displacements `dF_ij = eF_ij - e0_ij`.
6. For every named condition and geometry variant, calculate the displacement distribution,
   population direction, target projection, displacement coherence, residual/off-axis
   displacement, collateral measures, and per-stratum distributions.
7. Compare paraphrase directions and projections to the original and to run-manifest-frozen
   controls.
8. Apply the locked decision rule once; retain all null, unfavorable, and disagreement
   artifacts.

### Evidence schema and measurements

The evidence package contains:

- raw records in the shared schema and a complete pairing/attrition table;
- immutable condition texts and hashes;
- per-pair displacement vectors or provenance-bearing references;
- full target-projection distributions by condition and stratum;
- population directions and bootstrap stability as sample count increases;
- displacement-coherence statistics;
- residual/off-axis displacement and global displacement norms;
- paraphrase-to-original alignment and paraphrase agreement matrix;
- neutral/bleached empirical-null comparisons;
- run-manifest-frozen collateral measures, including verbosity, sentiment, refusal frequency,
  rhetorical structure, deference, hedging, answer length, and topic avoidance where the
  corresponding run-manifest-frozen instruments exist;
- geometry-variant comparison and any disagreement;
- frozen calibration/decision manifest and complete provenance.

No single scalar “funnel score” is emitted.

### Result

**PENDING — not run.**

### Analysis

**PENDING.** H0 and predicted behavior were recorded before experiment-layer implementation
or unit-test observation. Subsequent implementation/unit-test results are contract evidence
only and are not analyzed here. No 0A model output, embedding, or experimental statistic has
been observed or interpreted as funnel evidence.

---

## `SKM-FUNNEL-EXP0B` — Clause decomposition and reinforcing alignment

### Entry condition and hypothesis position

0B begins only after selecting a genuinely multi-clause candidate with constituent boundaries
frozen before output generation. An H1 result from 0A may motivate 0B, but it is not rewritten
as evidence for H2 or H3.

This experiment tests H0 against H2 and H3 as separate claims:

- **H0 — No reinforcing composition:** constituent displacements are no more positively
  aligned than the future run-manifest random/unrelated directive controls, and combination
  behavior is explainable by ordinary variation or the additive constituent prediction.
- **H2 — Reinforcing constituent alignment:** independently tested clauses produce a stable
  positive alignment structure in their displacement fields relative to the future
  run-manifest controls.
- **H3 — Compositional reinforcement:** a combined intervention produces positive
  target-axis interaction beyond the independently measured constituent prediction, without
  counting larger global/off-target displacement as improvement.

H2 does not imply H3. H3 is not inferred solely from vector-norm growth. Failure of H2/H3
does not invalidate an independently established H1.

### Conditions

For a three-clause funnel, the future run manifest would freeze the full design:

```text
baseline
F1
F2
F3
F1 + F2
F1 + F3
F2 + F3
F1 + F2 + F3
```

Also run the ADR-SKM-009 reinforcement controls:

- shuffled constituent combinations/orderings;
- random directive pairs;
- unrelated effective directives;
- deliberately opposed directive pairs;
- token/length-matched neutral controls.

If the candidate has a constituent count other than three, freeze the complete feasible
factorial or an explicitly justified subset before generation. No combination may be added or
removed after its output is inspected.

### Predicted observable behavior

If H2 holds:

1. independently measured constituent mean displacements form a reproducible positive block
   in the full Gram/cosine matrix;
2. the block is stable under prompt/seed resampling and exceeds the alignment structure of
   random, unrelated, and deliberately opposed directive controls;
3. the matrix remains available at full resolution, revealing dominant, antagonistic, or
   clustered constituents rather than hiding them in one mean cosine.

If H3 additionally holds:

1. observed combination target projection exceeds its independently measured additive
   prediction by a positive interaction residual under the locked empirical comparison;
2. that interaction survives the run-manifest-frozen population strata and bootstrap
   stability check;
3. residual/off-target displacement does not explain the apparent gain.

### Planned falsifier and decision-rule shape (not yet precommitted)

These clauses record intended falsifiers, not frozen numerical rules. The separate pilot (if
needed) and final run manifest must freeze all statistics, cutoffs, sample sizes, stability
requirements, and error control before constituent or combination outputs are observed. The
resulting frozen analysis then gives separate H2 and H3 verdicts:

- **Advance H2 for independent replication** only if the run-manifest-defined positive block
  in the complete constituent alignment matrix is stable under the locked
  bootstrap/sample-count procedure and exceeds the corresponding structure of random,
  unrelated, and opposed
  directive controls under the locked empirical-null decision rule.
- **Withhold/reject H2 under the future run manifest** if alignment is unstable,
  approximately null, negative/competing, attributable to one dominant clause rather than the
  run-manifest-defined shared block, or not distinguishable from directive controls. Those
  outcomes are banked as the
  measured constituent structure, not discarded.
- **Advance H3 for independent replication** only if the target-axis interaction residual
  between each run-manifest-defined observed combination and `d_additive = Σ d_k` is positive
  under the locked comparison, stable across the run-manifest-frozen population, and not
  replaced by global norm growth or collateral displacement.
- **Withhold/reject H3 under the future run manifest** when composition is additive,
  subadditive,
  unstable, off-axis, or driven by collateral/global shift. Approximately additive and
  antagonistic outcomes are informative results.
- **Not evaluable** applies if calibration, constituent power/stability, pairing, or attrition
  fails the frozen validity requirements; it cannot be repaired post hoc.

An H2 verdict permits only the claim that constituent **emission-space displacement fields**
are mutually aligned under the run-manifest-frozen instrument. An H3 verdict permits only
measured positive interaction along the run-manifest-frozen target dimension. Neither licenses
an internal vector, attractor, or recursive-reinforcement claim.

### Controlled method

1. Complete the separate pilot registration if calibration is needed; otherwise justify the
   exact externally specified values.
2. Freeze constituent segmentation, all combinations/orderings, controls, prompt population,
   seed schedule, target axis, instruments, analysis revision, and exact decision parameters
   in the versioned run manifest.
3. Generate all conditions over identical prompt/seed cells without adaptive regeneration.
4. Persist and embed outputs under the shared provenance and pairing rules.
5. Compute every constituent's paired displacement population and mean direction.
6. Retain the complete displacement Gram matrix `G_ij = <d_i, d_j>` and calibrated cosine
   matrix `A_ij = cos(d_i, d_j)` for each run-manifest-frozen geometry variant.
7. Bootstrap matrix stability as sample count increases and compare the full structure with
   random, unrelated, and opposed directive controls.
8. For every run-manifest-frozen combination, compute observed displacement, additive
   prediction, target-axis interaction residual, residual/off-axis displacement, and
   collateral measures.
9. Apply H2 and H3 decision rules separately; retain additive, null, subadditive,
   antagonistic, and geometry-disagreement outcomes.

### Evidence schema and measurements

The evidence package contains:

- raw records in the shared schema with exact ordered `constituents` and pairing/attrition;
- immutable constituent/combination/control texts and hashes;
- per-condition paired displacement populations and mean directions;
- complete Gram and calibrated-cosine matrices, not only aggregate cosine;
- bootstrap matrix-stability traces as sample count increases;
- random/unrelated/opposed-directive control matrices;
- target projection for each constituent and combination;
- additive prediction `Σ d_k` for each combination;
- observed-minus-additive interaction vector and its target-axis projection;
- global norm, residual/off-axis displacement, and run-manifest-frozen collateral dimensions;
- per-stratum consistency distributions;
- all run-manifest-named geometry variants, including a whitening comparison if selected,
  and their disagreement report;
- frozen calibration/decision manifest and complete provenance.

No single scalar “funnel score” is emitted, and no average cosine replaces the complete
alignment matrix.

### Result

**PENDING — not run.**

### Analysis

**PENDING.** H0 and predicted behavior were recorded before experiment-layer implementation
or unit-test observation. Subsequent implementation/unit-test results are contract evidence
only and are not analyzed here. No 0B model output, embedding, Gram matrix, composition
residual, or experimental statistic has been observed or interpreted as funnel evidence.

---

## Outcome banking and follow-on boundaries

For both experiments, publish or bank the full evidence package regardless of verdict.
Specifically retain:

- null, weak, contradictory, or instrument-disagreeing outcomes;
- paraphrase-specific effects (“magic string” evidence);
- independent, competing, redundant, additive, subadditive, and off-axis constituent
  structures;
- invalid/underpowered runs with the reason they were not evaluable.

No 0A/0B result automatically promotes a claim to H4 attractor-like behavior or H5 recursive
reinforcement. Experiment 0C requires its own preregistration before multi-turn output is
observed. Autonomous funnel synthesis remains out of scope until 0A and 0B yield independently
replicable, boringly repeatable measurements; sk-mcp remains the measurement instrument, not
the optimizer.
