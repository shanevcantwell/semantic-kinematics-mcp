# Draft protocol / preregistration scaffold: “Literally” as a contextual operator

**Stable handle:** `SKM-LIT-OP-001`

**Status:** draft protocol / preregistration scaffold; **not a completed preregistration and not an empirical result**

**Scaffold recorded:** 2026-09-23 (US/Mountain)

**Related decision:** [`ADR-SKM-009`](../ADRs/proposed/adr-skm-009-funnel-measurement-profile.md)

**Related experiment program:** [`SKM-FUNNEL-EXP0`](./2026-09-23-funnel-experiments-0a-0b-preregistration.md)

This document records H0 and the predicted behavior for this experiment before any
stimulus embeddings, model outputs, response embeddings, or experimental statistics from
`SKM-LIT-OP-001` are generated or inspected. It does not freeze the corpus, model,
instrument configuration, sample size, statistics, cutoffs, or decision rules. A separately
versioned run manifest must freeze those choices before a run may be called preregistered or
registered.

---

## Purpose

The word *literally* is a useful probe because its function depends on the proposition and
reading it modifies. In one context it marks an assertion as factual or non-figurative; in
another it is a bleached scalar intensifier. The object of interest is therefore not the
location of the token *literally* in embedding space. It is the context-conditioned
transformation associated with adding that modifier to a complete, matched stimulus or with
observing the resulting change in a matched model response.

The experiment asks four progressively stronger questions:

1. Does *literally* have a stable regime-dependent interaction beyond generic modifier and
   lexical-frame effects?
2. Is that interaction consistent with contextual selection among operator-like functions?
3. In factual contexts, does its displacement align with actuality/non-metaphorical
   operators, while in bleached contexts it aligns with scalar intensifiers?
4. In model behavior, does factual *literally* suppress the accessibility of figurative
   interpretations strongly and selectively enough to count as a small, controlled
   **micro-funnel**?

The design is a sandbox for the paired-displacement and alignment measurements in
ADR-SKM-009 and `SKM-FUNNEL-EXP0`. It is not itself Funnel 0A or 0B.

---

## Claim and evidence boundary

### Layer A — Direct stimulus-embedding geometry

Embed complete, matched stimulus strings and compare their geometry. This layer is a
**lexical/compositional diagnostic only**. It can show that an embedding instrument assigns
a regime-dependent displacement to complete strings, and it can compare that displacement
with other modifiers. It cannot establish that a generating model changed its interpretation,
that an interpretation became less accessible, or that a behavioral funnel exists.

### Layer B — Emission-space behavioral geometry

Give complete matched stimuli to a frozen generating model under a frozen elicitation task,
then embed the complete matched responses. Pairing is by stimulus frame, regime, modifier,
model checkpoint, invocation settings, and seed where supported. This is the primary geometric
evidence layer for contextual function selection and the micro-funnel/accessibility claim.

Direct behavioral labels and interpretation-choice frequencies are required alongside Layer B.
For example, a frozen task may elicit an interpretation, continuation, paraphrase, entailment,
or forced choice whose factual versus figurative reading can be scored independently. Response
embeddings supplement those behavioral observables; they do not replace them.

Layer A and Layer B results must remain separately identified in every artifact and claim. A
positive Layer A result with a null Layer B result is evidence about the embedding instrument's
stimulus geometry, not about model behavior.

### Explicit prohibition on internal claims

External stimulus or response embeddings do not expose attention weights, hidden states,
activations, gradients, or token-to-token causal interactions. Results from this protocol must
not be called **“second-order cross-attention,”** an attention interaction, or evidence of an
internal transformer mechanism.

Permitted language is limited to:

- **stimulus-embedding interaction** for Layer A; and
- **emission-space interaction** or **behavioral interaction** for Layer B.

Any internal cross-attention claim would require a separately designed internal-state or
causal-intervention experiment.

---

## Experimental object and factorial interaction

Let:

- `i` identify a matched lexical-frame block whose regime realizations are linked by one
  frozen compatibility and complete-string template rule;
- `r` identify a usage regime;
- `m` identify a modifier condition;
- `j` identify a matched model run/seed for Layer B;
- `x(i,r,m)` be the **complete grammatical stimulus** for that cell;
- `S(i,r,m) = E(x(i,r,m))` be its Layer A embedding; and
- `B(i,r,m,j) = E(y(i,r,m,j))` be the embedding of the complete Layer B response.

For Layer A, a matched block `b` is `i`; for Layer B, it is the matched frame-run block
`b = (i,j)`. The seed index is omitted for Layer A. For modifier `m` and registered regimes
`r1` and `r2`, first compute the four-cell interaction *within each block*:

```text
D_Z(m; r1,r2,b)
    = [Z(i,r1,m,j) - Z(i,r1,none,j)]
    - [Z(i,r2,m,j) - Z(i,r2,none,j)]
```

A block enters this contrast only if it supplies all four complete, usable cells under the
same frozen compatibility/template rule: `(r1,m)`, `(r1,none)`, `(r2,m)`, and `(r2,none)`.
A block missing any member is excluded in full from that contrast. The evidence package
must show eligible-block counts, included-block counts, exclusions and reasons by regime and
modifier, rather than silently changing denominators or regenerating a missing member. A block
may therefore be eligible for one registered modifier contrast and ineligible for another.

Only after those within-block interactions are formed is the geometric estimand summarized
over blocks and, for Layer B, matched runs:

```text
I_Z(m; r1,r2)
    = sum[b in C(m,r1,r2)] w_b D_Z(m; r1,r2,b)
      / sum[b in C(m,r1,r2)] w_b
```

`C(m,r1,r2)` is the four-cell common-support set, and the weighting rule `w_b` must be fixed in
the future manifest before confirmatory outputs are generated. Equal-block weighting, or a
specified frame/run hierarchy, remains to be chosen; weights may not be adapted to observed
geometry. Complete per-block interactions remain available, and resampling/inference must
preserve frame and run clustering. Independently averaging the two regime populations and
then subtracting those averages is not this estimand and is prohibited, even when their sample
counts happen to match.

The primary geometric interaction uses `m = literally` and separately registered
factual-versus-bleached regime pairs. Here `bleached` must be resolved in the future manifest rather
than silently pooled. At minimum, idiomatic/metaphorical-target and colloquial-intensifier
strata are analyzed separately; a manifest may additionally register a justified aggregate
while retaining both components. The same matched-block estimand is calculated for every
registered control modifier, supporting a complete operator-by-regime comparison rather than
a privileged one-word test.

This difference-of-differences captures the intended interaction residual: the additional
effect of *literally* depends on the proposition/usage regime to which it applies. It does so
using complete cells. The protocol does **not** require embedding a dangling `C + "literally"`
fragment, subtracting an incomplete string, or treating a modifier in isolation as a semantic
baseline. Avoiding incomplete fragments removes a grammatical-completeness and prompt-shape
confound from the interaction estimate.

If a future design cannot construct four-cell common support across regimes, its manifest must
instead prespecify a hierarchical estimator, its exchangeability/partial-pooling assumptions,
weights, and diagnostics. The resulting quantity must be labeled a **cross-stratum
observational contrast**, not a causal or contextual interaction, and cannot satisfy an H1–H3
interaction claim under this protocol.

---

## Usage strata and illustrative seeds

The future corpus must include these strata:

1. **Factual literal use** — the proposition denotes an event that can be asserted as having
   occurred non-figuratively. Illustrative seed: “He literally broke the glass.”
2. **Metaphorical/idiomatic target** — the predicate invites a conventional figurative or
   idiomatic reading. Illustrative seed: “He literally exploded with anger.”
3. **Colloquial intensifier use** — *literally* is naturally read as emphatic rather than as a
   correction toward a physical reading. Illustrative seed: “I literally died laughing.”
4. **Explicit literal-versus-figurative contrast** — the discourse explicitly makes the
   competing reading salient. Illustrative seed: “Not figuratively—he literally broke the
   wall.”

These sentences are **illustrative seeds, not a frozen corpus**. The run manifest must define
multiple lexical frames, verbs, topics, syntactic realizations, and matched complete
no-modifier/control versions. In the explicit-contrast stratum, removing a modifier must not
leave an ungrammatical contrast; the baseline and control frames must be rewritten by a frozen,
meaning-preserving template rule and matched at the full-cell level.

Lexical-frame, verb, and topic families must be assigned to development and holdout partitions
before confirmatory outputs are observed. A result carried by one verb (for example, *explode*
or *die*), one idiom, one topic, or one frame is not a stable contextual interaction.

---

## Modifier conditions and control roles

Every registered frame must use whichever complete, grammatical cells are licensed by the
future manifest. Required conditions are:

- **baseline:** no modifier;
- **probe:** `literally`;
- **actuality/non-metaphorical operator:** `actually` and any additional operator admitted by
  a frozen, independently justified control set;
- **opposite-reading operators:** `figuratively`, `metaphorically`;
- **scalar intensifiers:** `seriously`, `really`, `absolutely`, `totally`;
- **matched lexical/frame controls:** frozen alternatives that match syntax, register,
  frequency, token length, or character length where feasible without asserting the target
  semantic function.

These groups are not interchangeable. `Actually` probes actuality/correction; `figuratively`
and `metaphorically` explicitly favor the competing reading; `seriously`, `really`,
`absolutely`, and `totally` probe scalar emphasis. Each modifier must be tested within complete,
licensed frames. Ungrammatical or semantically incoherent modifier-frame combinations are not
silently retained or dropped: compatibility rules and exclusions must be frozen before data
generation, and attrition must remain visible.

The protocol also requires paraphrases or constructional variants of the relevant operator
functions. This separates a contextual function from an exact-word or “magic string” effect.
No claim depends on assuming these words are perfect synonyms.

---

## Hypothesis ladder and predictions

### H0 — No stable contextual interaction

After complete-cell pairing and empirical controls, the factual-versus-bleached interaction
for *literally* is not stable across lexical frames, verbs, topics, paraphrases, and held-out
families. Apparent effects are explainable by ordinary lexical frequency, modifier length,
frame compatibility, prompt sensitivity, run variation, or the embedding instrument.

### H1 — Contextual function selection

The displacement induced by *literally* differs reproducibly by usage regime. Factual and
explicit-contrast contexts show a different stable transformation from idiomatic and
colloquial-intensifier contexts, beyond the corresponding no-modifier and matched-control
differences.

Predicted observable behavior:

- the matched-block, four-cell difference-of-differences is stable under registered
  resampling and on lexical-frame/verb/topic holdouts;
- the effect appears across multiple constructional variants rather than only the exact token
  sequence in an illustrative seed; and
- the complete operator-by-regime matrix shows regime structure not reducible to global
  displacement magnitude.

A Layer A result can support only a stimulus-geometry version of H1. A behavioral H1 claim
requires Layer B and direct behavioral evidence.

### H2 — Regime-specific operator alignment

In factual and explicit-contrast regimes, the *literally* displacement is predicted to align
more closely with registered actuality/non-metaphorical operator displacements than with
scalar-intensifier displacements. In bleached/colloquial regimes, it is predicted to align more
closely with registered scalar-intensifier displacements than with actuality-operator
displacements. Opposite-reading operators provide a directional and structural contrast; no
sign or magnitude threshold is asserted by this scaffold.

The complete alignment matrices must be retained. One average cosine may not replace evidence
of clusters, antagonistic operators, frame-dependent reversals, or one dominant control.

### H3 — Behavioral micro-funnel / accessibility suppression

The future run manifest must select and freeze the exact direct interpretation outcome before
confirmatory generation. Admissible generic forms include a binary figurative-accessible
indicator, an ordinal figurative-accessibility score, or a forced-choice probability assigned
to the figurative reading. It must also freeze the coding direction, elicitation, aggregation,
and handling of ambiguous or invalid responses. This scaffold does not choose among those
outcomes or freeze a numerical value.

Let `A(i,r,m,j)` denote the selected direct outcome, coded so that larger values mean greater
figurative accessibility. For every prespecified target regime `r1`, comparison regime `r2`,
and control `c`, compute the direct behavioral interaction within the same complete
frame-run block:

```text
D_A(literally,c; r1,r2,b)
    = [A(i,r1,literally,j) - A(i,r1,c,j)]
    - [A(i,r2,literally,j) - A(i,r2,c,j)]

H3_A(literally,c; r1,r2)
    = sum[b in C_A(literally,c,r1,r2)] w_b D_A(literally,c; r1,r2,b)
      / sum[b in C_A(literally,c,r1,r2)] w_b
```

Each behavioral common-support set contains only blocks with all four members. Weighting and
attrition follow the same frozen rules as the geometric estimand, and uncertainty estimation
must preserve lexical-frame and run/seed clustering. The manifest must prespecify contrasts
with `c = none`, with the designated scalar-intensifier control or controls, and with the
designated matched lexical control or controls; it must not select the favorable control after
observing results. Factual and explicit-contrast target regimes and each bleached comparison
stratum must likewise be named rather than pooled opportunistically.

H3 predicts the registered interaction direction consistent with selectively lower figurative
accessibility under factual or explicit-contrast *literally*, relative to all three control
roles. Geometry may supplement this result as a target-directed emission-space displacement
toward grounding/actuality with bounded collateral movement, but it cannot substitute for the
direct outcome. A factual-only modifier main effect such as
`A(factual,literally) - A(factual,none)` cannot establish regime-selective micro-funnel
suppression: it lacks the matched comparison-regime difference and the required control
contrasts.

This is a micro-funnel claim because the modifier selectively constrains an interpretation
family. It is not supported by generic response homogenization, answer-length change, lexical
echoing of the prompt, or a large off-axis embedding shift.

### Ladder discipline

Failure of H3 does not invalidate H2 or H1. Failure of H2 does not invalidate a stable H1
interaction. A Layer A finding does not advance a Layer B hypothesis. None of H1–H3 implies an
internal attention head, cross-attention interaction, hidden-state direction, or model-internal
operator vector.

---

## Planned measurements

All measurements are planned estimand shapes, not frozen numerical settings, exact outcome
choices, or decision thresholds.

### Paired displacement and interaction

For each evidence layer and registered modifier-by-regime contrast:

- retain the paired modifier displacements that constitute each eligible four-cell block;
- compute the within-block interaction before any population summary;
- apply the frozen weights only to the common-support set, preserving frame/run clustering;
- calculate factual-versus-bleached and other manifest-registered matched-block
  difference-of-differences;
- report per-block interaction vectors, their distributions, stability, and visible attrition
  rather than only population norms; and
- do not construct the interaction by subtracting independently averaged regime populations.

### Complete operator-by-regime geometry

Retain full Gram and calibrated-cosine matrices over every registered modifier-by-regime
population direction. Any cross-regime contrast must estimate those entries from its same
four-cell common-support blocks with the frozen weights; regime-specific matrices over broader
populations are descriptive and cannot replace the matched interaction. Report matrices for
Layer A and Layer B separately. Compare matrix structure under bootstrap resampling and as the
number of matched blocks increases.

### Independently registered target axes

Before confirmatory output generation, register independently sourced positive/negative
examples for at least:

- a **grounding/actuality versus figurative-reading** axis; and
- an **intensification versus non-intensification** axis.

Project paired populations and interaction residuals onto both axes using frozen calibration
and empirical nulls. Axis examples may not be selected from the confirmatory corpus or revised
after viewing condition results. Report disagreement across registered geometry variants
rather than selecting one retrospectively.

### Residual and off-axis movement

For every target projection, retain the orthogonal/residual component, total displacement,
and registered collateral observables. At minimum, the future manifest must decide how to
track lexical echo, response length, verbosity, sentiment, hedging, refusal, register,
rhetorical structure, and topic drift. A larger global shift is not target efficacy.

### Behavioral accessibility

Layer B must include the direct behavioral outcome selected from the generic forms in H3. The
run manifest must define the exact outcome, elicitation and coding procedure, admissible
labels, coding direction, rater/model blinding, adjudication, and statistical treatment. It
must also freeze the no-modifier, intensifier, and matched lexical control contrasts and apply
the H3 within-block four-cell estimand with visible attrition and frame/run clustering.
Geometry and direct behavior are reported jointly and may disagree.

### Generalization and stability

Report by development/holdout status and by lexical frame, verb, topic, usage stratum,
constructional variant, and seed where applicable. Bootstrap stability is required for:

- interaction direction and target projection;
- operator-by-regime alignment structure; and
- behavioral-label or interpretation-choice effects.

The resampling unit, hierarchy, statistic, stopping rule, and stability criterion remain to be
frozen. No sample count or cutoff is implied here.

---

## Nulls, controls, and falsifiers

The future run manifest must define measured empirical controls that include:

- complete no-modifier cells;
- matched lexical and syntactic frames;
- modifier-frequency and modifier-length matching where feasible;
- paraphrases or constructional variants;
- actuality, opposite-reading, and scalar-intensifier control groups;
- shuffled or mismatched modifier-proposition pairings that preserve complete-string quality
  under a frozen construction rule;
- empirical background/null populations appropriate to each embedding instrument and target
  axis; and
- matched model-run controls for Layer B.

There is **no random-vector fallback**. A theoretical isotropic-vector comparison may not
silently substitute for a missing empirical corpus/background null.

The hypotheses are weakened or falsified under the future locked decision procedure by, among
other outcomes:

- failure to replicate across held-out frames, verbs, or topics;
- the same geometry for *literally* across all regimes;
- equal or stronger interaction for lexical/frequency/length controls;
- dependence on one illustrative sentence, idiom, or exact wording;
- operator alignment inconsistent with the registered regime-specific prediction;
- geometric change without corresponding direct behavioral change for a behavioral claim;
- direct behavioral change attributable to response length, lexical echo, or task demand; or
- sensitivity to instrument/preprocessing choices that does not survive independent
  replication.

Null, conflicting, and not-evaluable outcomes remain part of the evidence package.

---

## Current instrument planning context

At scaffold-writing time, the candidate embedding instrument available through MCP is:

```text
model family: nvidia/NV-Embed-v2
output dimensionality: 4096
service state: loaded through MCP
```

This is planning context only. The exact checkpoint revision, artifact hashes, preprocessing,
query/document instructions, pooling, normalization, dtype, batching, centering, calibration,
and empirical-null references remain **unfrozen**. The loaded state is not an experimental
observation and does not register this instrument for a run.

Forcing the same checkpoint to load or compute in FP32 would be a **precision-sensitivity
variant**, not an independent embedding instrument. Casting lower-precision stored weights to
FP32 does not restore native precision absent from those weights. Independent-instrument
replication requires a genuinely separate embedding model/checkpoint lineage, not a dtype
variant of the same checkpoint.

---

## Future run-manifest freeze requirements

Before generating, embedding, or inspecting confirmatory cells, a versioned run manifest must
freeze:

1. the exact research phase and which hypotheses/evidence layers it tests;
2. stimulus corpus version, complete cell texts, frame-generation rules, regime labels,
   compatibility rules, four-cell common-support construction, exclusions, and hashes;
3. development and holdout partitions for lexical frames, verbs, topics, and constructional
   variants;
4. all modifier groups, paraphrases, matched controls, and shuffled/mismatched pairing rules;
5. generating model/checkpoint, prompt wrapper, decoding/invocation parameters, seed schedule,
   and pairing/attrition policy for Layer B;
6. elicitation tasks, the exact direct behavioral outcome and coding direction, required
   no-modifier/intensifier/lexical-control contrasts, coder/rater protocol, blinding, and
   adjudication;
7. each embedding instrument's exact revision, preprocessing, instructions, pooling,
   normalization, dtype, batching, and artifact hashes;
8. centering/calibration procedures, empirical background nulls, registered geometry variants,
   and independent target-axis artifacts;
9. analysis code revision and full provenance/data schema;
10. matched-block estimators, fixed weights, frame/run clustering, attrition reporting,
    resampling units, bootstrap procedure, sample sizes, cutoffs, multiplicity and error-control
    policy, decision rules, stopping rules, and not-evaluable criteria; and
11. the exact permitted claim associated with each possible result.

No numerical value, cutoff, effect size, sample size, or quorum is invented by this scaffold.
If a null-only calibration pilot is needed to choose any such value, it must be separately
registered, use data disjoint from confirmatory cells, and specify in advance the deterministic
rule by which pilot outputs set confirmatory values. Confirmatory conditions may not be
inspected during pilot calibration.

Every raw record must carry enough provenance to reconstruct its complete stimulus, evidence
layer, regime, modifier, frame family, holdout status, model-run pairing, response reference,
embedding reference, instrument configuration, analysis revision, four-cell block membership,
and contrast-specific eligibility. Missing block members remain visible; they are not silently
regenerated or substituted.

---

## Anti-Goodhart and holdout boundary

This protocol is diagnostic, not an optimization loop. If wording, frames, target axes, or
elicitation tasks are developed iteratively, that work occurs only on the development
partition. Holdout stimuli, responses, labels, embeddings, and per-cell errors remain hidden
until the manifest, analysis, and decision rules are frozen.

The following boundaries apply:

- target axes and calibration cannot be revised to improve observed condition separation;
- controls cannot be dropped because they align inconveniently;
- Layer A may not be used to select only stimuli expected to succeed in Layer B holdout;
- exact holdout responses should remain unavailable to any future synthesizer where practical;
- all registered matrix cells and geometry variants are reported, including disagreements;
- direct behavioral outcomes remain visible even when they conflict with embedding geometry;
- any adaptive follow-up receives a new handle/version and a fresh holdout; and
- a result confined to NV-Embed-v2 remains instrument-bounded until independently replicated.

No single scalar “literally score” is defined. The evidence surface retains interaction,
operator alignment, target projections, off-axis movement, behavioral accessibility, and
holdout transfer separately.

---

## Result

**PENDING — not run.**

No stimulus embeddings, model calls, response embeddings, or experimental statistics have
been generated or inspected for `SKM-LIT-OP-001` under this scaffold.

## Analysis

**PENDING.**

H0 and the predicted behavior above are recorded before observation of this experiment. This
section must not be populated until a separately versioned run manifest has been frozen and
its evidence package has been produced.

---

## Carry-forward criterion for Funnel 0A/0B

This micro-funnel design would justify carrying its operator-style structure into broader
Funnel 0A/0B work only if a future locked run produces a **boringly repeatable Layer B effect**:
a complete-cell contextual interaction that survives lexical-frame, verb, topic, paraphrase,
and empirical-null controls; transfers to held-out families; agrees with direct interpretation
behavior; shows the predicted regime-sensitive operator alignment or clearly characterized
alternative structure; and keeps residual/collateral displacement visible.

That outcome would justify testing whether larger directives similarly suppress families of
interpretations or continuations and whether separately measured clauses align. It would not,
by itself, establish a general funnel, compositional reinforcement, an internal vector, or an
attention mechanism. A Layer A-only success, a magic-string effect, or geometry without direct
behavioral agreement would not justify that carry-forward.
