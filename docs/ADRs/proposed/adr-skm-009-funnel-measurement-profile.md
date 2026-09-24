# ADR-SKM-009: Measure and synthesize reinforcing directive funnels

**Status:** proposed
**Date:** 2026-09-23 (US/Mountain)
**Author:** shanevcantwell, with pi as documentation collaborator
**Related:**
- **Direct lineage refined and extended here** — [`docs/ADRs/proposed/adr-skm-008-functional-direction-probe-generalized-axis-source.md`](./adr-skm-008-functional-direction-probe-generalized-axis-source.md) (functional directions as generalized axis sources over the existing projection contract). This ADR adds population-level intervention measurement and constituent-composition geometry to ADR-SKM-008; it does not supersede it.
- **Position geometry and empirical calibration** — [`docs/ADRs/proposed/ADR-001-referential-axis-alignment.md`](./ADR-001-referential-axis-alignment.md) (centered/calibrated directional measurement and empirical corpus nulls).
- **Directional projection math** — [`docs/ADRs/proposed/ADR-SKMCP-0001-directional-projection-primitive.md`](./ADR-SKMCP-0001-directional-projection-primitive.md) (signed projection, orthogonal residual, measured null, and single-embedder discipline).
- **Measurement artifact/tool-family contract** — [`docs/ADRs/proposed/ADR-SKMCP-0002-bearing-analysis-tool-contract.md`](./ADR-SKMCP-0002-bearing-analysis-tool-contract.md) (build-validated artifact, fast consume path, provenance, and falsification-shaped confidence).
- **Embedder-specific geometry constraint** — [`docs/ADRs/proposed/adr-skmcp-0004-affect-geometry-targets-nv-embed.md`](./adr-skmcp-0004-affect-geometry-targets-nv-embed.md) (cross-instrument sensitivity and the requirement to measure rather than assume geometry transfer).
- **Adjacent programs** — prompt-prix (external candidate synthesis/harness); LNF falsification pipeline (independent holdout and claims gate).

**Supersedes:** — (additive; refines and adds to ADR-SKM-008 without replacing it or any existing measurement primitive).
**Superseded by:** TBD.

---

## Context

Small contextual directives can produce behavioral effects far larger than their
literal token count or apparent scope.

A short instruction such as:

> Respect the precise definitions when using terms of art.

may suppress a large family of downstream behaviors without enumerating those
behaviors individually.

Call such an intervention a **funnel**:

> A compact contextual intervention that systematically constrains or redirects
> a population of otherwise varied response trajectories.

Funnels are currently recognizable qualitatively but not operationally measurable.
That leaves several possibilities confounded:

- a genuine broad behavioral constraint;
- ordinary prompt sensitivity;
- a memorable phrasing effect;
- multiple independent instructions producing unrelated changes;
- several instructions reinforcing the same behavioral direction;
- global style collapse mistaken for targeted constraint;
- a multi-turn effect that strengthens because generated output itself reinforces
  the original constraint.

Recent work on representation steering, preference-vector geometry, gradient
alignment, low-dimensional RL adaptation, and coupled behavioral directions gives
a concrete reason to test a stronger hypothesis:

> Multiple contextual interventions may induce measurably aligned behavioral
> displacement fields, and their composition may reinforce a common behavioral
> direction rather than merely add unrelated prompt effects.

This ADR does **not** assume that output-embedding geometry is identical to
internal model geometry.

The initial measurable object is behavior in emission space.

Internal activations, gradients, or model-state geometry may later be used as an
independent instrument to test whether the externally observed geometry has an
internal correlate.

---

## Decision

Add a first-class **funnel measurement profile** to sk-mcp.

sk-mcp will measure the effect of contextual interventions across populations of
matched model runs.

An external orchestrator such as prompt-prix or pi may use those measurements to
search autonomously for candidate funnels.

The architecture remains:

```text
synthesizer / harness
    proposes interventions
    generates matched populations
    maintains search state
    selects candidates

            ↓

sk-mcp
    embeds
    calibrates
    measures displacement
    compares geometry
    evaluates nulls
    emits provenance-bearing measurements

            ↓

falsifier / holdout gate
    tests robustness
    detects Goodharting
    rejects unsupported claims
```

**sk-mcp does not become the optimizer.**

The optimizer does not get to redefine the measurement instrument that evaluates
its current search.

This decision **refines and adds to ADR-SKM-008** by applying its functional-direction
and calibrated-projection lineage to matched intervention populations, constituent
alignment, and composition. ADR-SKM-008 remains in force and is not superseded.

---

## Operational definition

For prompt `p_i`, run/seed `j`, baseline context `C`, and candidate funnel `F`:

```text
y0_ij = model(C + p_i, seed_j)
yF_ij = model(C + F + p_i, seed_j)
```

Using registered embedding instrument `E`:

```text
e0_ij = E(y0_ij)
eF_ij = E(yF_ij)
```

Define paired displacement:

```text
dF_ij = eF_ij - e0_ij
```

The population:

```text
D_F = { dF_ij }
```

is the primary observable funnel effect.

A funnel is not represented by one scalar.

It is represented by a population-level geometric profile.

---

## Reinforcing alignment

For a funnel containing constituent directives:

```text
F = {F1, F2, ... Fn}
```

measure each constituent independently:

```text
d_k = mean(E(y | F_k) - E(y | baseline))
```

Construct the displacement Gram matrix:

```text
G_ij = <d_i, d_j>
```

or calibrated cosine equivalent:

```text
A_ij = cos(d_i, d_j)
```

This distinguishes several cases.

### Independent effects

```text
cos(d_i, d_j) ≈ 0
```

The clauses affect different behavioral dimensions.

### Competing effects

```text
cos(d_i, d_j) < 0
```

The clauses partially cancel one another.

### Reinforcingly aligned effects

```text
cos(d_i, d_j) > 0
```

The clauses independently push behavior in compatible directions.

A strong positive block structure in the Gram matrix is evidence that apparently
different directives share a common behavioral direction.

**Constituent alignment is a first-class measured object**, distinct from target-axis
efficacy and retained at matrix resolution rather than immediately collapsed to one
score.

“Reinforcingly aligned” is an operational term in this ADR, not a claim that the
model possesses a single literal internal vector corresponding to the funnel.

---

## Composition and superadditivity

Measure the complete funnel:

```text
d_F = mean(E(y | F1 + F2 + ... + Fn) - E(y | baseline))
```

Compare it against the independently measured constituent prediction:

```text
d_additive = Σ d_k
```

Three regimes matter.

### Subadditive

```text
||d_F|| < ||d_additive||
```

Constituents interfere or saturate.

### Approximately additive

```text
d_F ≈ d_additive
```

The combined effect is explainable by independent contributions.

### Superadditive

The combined intervention produces more target-directed effect than predicted
from constituent interventions alone.

This should not be defined only as vector norm growth.

Superadditivity should be measured along registered target dimensions while also
tracking off-target displacement.

A candidate does not count as improved merely because it causes a larger global
embedding shift.

---

## Funnel measurement profile

### 1. Directional efficacy

Measure projection onto a preregistered behavioral target axis:

```text
efficacy(F) = projection(D_F, v_target)
```

Use existing centered/calibrated sk-mcp directional machinery and empirical nulls.

### 2. Population consistency

Determine whether the effect survives across:

- prompts;
- seeds;
- task families;
- topic strata;
- conversation positions.

Report the effect distribution.

Do not allow a few extreme examples to stand in for a population effect.

### 3. Displacement coherence

Measure whether paired displacement vectors themselves occupy a common direction.

This asks:

> Did the intervention create a reproducible transformation?

independently from:

> Was that transformation the one we intended?

These questions must remain separate.

### 4. Constituent alignment

For multi-clause funnels, calculate pairwise and population-level alignment among
constituent displacement fields.

Retain the complete alignment matrix.

Do not collapse it immediately into a single average cosine.

Potential structure such as clusters, antagonistic components, or one dominant
directive may be more informative than the aggregate.

### 5. Selectivity / collateral displacement

Decompose funnel displacement into:

```text
intended component
residual component
```

Measure whether the intervention also changes unrelated properties such as:

- verbosity;
- sentiment;
- refusal frequency;
- rhetorical structure;
- deference;
- hedging;
- answer length;
- topic avoidance.

Residual displacement is a result, not nuisance variance.

### 6. Robustness to wording

Generate semantically similar intervention variants.

A stable funnel should survive at least some paraphrasing.

This separates:

```text
constraint-level effect
```

from:

```text
magic-string effect
```

Surface-form invariance is evidence for, but not proof of, a more general
behavioral constraint.

### 7. Placement robustness

Vary funnel position:

- system context;
- developer-equivalent context where available;
- immediately before task;
- distant preceding context;
- repeated vs single insertion.

Measure how geometry changes with contextual placement.

---

## Attractor hypothesis

A funnel may do more than translate responses along a direction.

It may reduce variance across heterogeneous starting conditions.

Let baseline response embeddings be:

```text
X0 = {e0_i}
```

and funnel-conditioned embeddings:

```text
XF = {eF_i}
```

Measure dispersion relevant to the target behavior:

```text
dispersion(X0)
dispersion(XF)
```

A funnel exhibits **attractor-like behavior in emission space** when it both:

1. moves heterogeneous response populations toward a reproducible behavioral
   region; and
2. contracts variance along behaviorally relevant dimensions.

This must be distinguished from generic output homogenization.

A funnel that simply makes every answer stylistically similar is not evidence of
a useful attractor.

Residual dimensions and unrelated task performance must therefore remain visible.

---

## Recursive reinforcement

Multi-turn interaction creates a stronger hypothesis.

A model's generated output becomes part of its subsequent context:

```text
C_t --F--> y_t

C_(t+1) = C_t + y_t
```

If `y_t` itself adds context that points in the same measured behavioral direction
as `F`, the system may create positive semantic feedback.

Let:

```text
v_F = registered funnel direction
```

and measure turn-specific displacement:

```text
d_t
```

Define directional amplitude:

```text
a_t = <d_t, v_F>
```

Then test whether:

```text
a_(t+1) > a_t
```

systematically under matched conditions.

A stronger model approximates local behavioral dynamics as:

```text
d_(t+1) ≈ A d_t
```

If:

```text
A v_F ≈ λ v_F
```

then `λ` gives an operational measure of persistence or reinforcement.

Interpretation:

```text
0 < λ < 1
    effect persists but decays

λ ≈ 1
    approximately stable persistence

λ > 1
    self-reinforcing behavioral mode

λ < 0
    oscillatory / compensatory response

A v_F largely off-axis
    semantic drift rather than reinforcement
```

This is an **emission-space dynamical measurement**.

It must not be described as an internal transformer eigenvector without an
independent internal-state experiment.

---

## Hypothesis ladder

Experiments should distinguish progressively stronger claims.

### H0 — No funnel

Observed differences are ordinary prompt/run variation.

### H1 — Directional funnel

The intervention produces reproducible target-directed displacement.

### H2 — Reinforcing constituent alignment

Multiple independently tested funnel clauses produce positively aligned
displacement fields.

### H3 — Compositional reinforcement

The combined funnel produces a stronger target-directed effect than constituent
effects predict independently.

### H4 — Attractor-like behavior

The intervention causes heterogeneous responses to converge toward a common
behavioral region while preserving unrelated variation.

### H5 — Recursive reinforcement

In multi-turn use, generated outputs reinforce the same measured behavioral mode,
producing persistent or increasing directional amplitude.

Failure of H(n) does not invalidate H(n-1).

Do not promote evidence upward through this ladder automatically.

---

## Nulls and controls

Every experiment requires measured controls.

At minimum:

- no-funnel baseline;
- token/length-matched neutral instruction;
- paraphrased funnel variants;
- semantically bleached variants where feasible;
- shuffled constituent combinations;
- constituent directives tested independently;
- empirical corpus/background null appropriate to the embedding instrument.

Where measuring reinforcement:

- random directive pairs;
- unrelated effective directives;
- deliberately opposed directive pairs.

Where measuring recursion:

- identical multi-turn structure without funnel;
- funnel removed after initial turn;
- funnel retained but prior assistant output excluded where architecture permits.

No silent fallback to theoretical random-vector nulls.

---

## Autonomous funnel synthesis

An external synthesizer may search over funnel candidates using:

```text
propose
    ↓
generate matched population
    ↓
measure
    ↓
falsify
    ↓
retain / reject
    ↓
mutate / recombine
    ↓
repeat
```

Candidate operations may include:

- addition;
- deletion;
- paraphrase;
- compression;
- clause splitting;
- clause recombination;
- ordering changes;
- replacement by semantically different constraints.

The synthesizer should receive measurement results, not raw holdout answers when
avoidance is practical.

---

## Search objective

Do not create a single scalar “funnel score.”

The search surface should retain at least:

```text
target efficacy
population consistency
constituent alignment
compositional gain
collateral displacement
paraphrase robustness
placement robustness
holdout transfer
```

For recursive experiments also retain:

```text
persistence
estimated λ
off-axis drift
```

Candidate selection should use gates or Pareto comparison rather than hidden
scalar weighting.

---

## Anti-Goodhart requirements

Autonomous synthesis directly optimizes against a measurement instrument.

Therefore:

1. Freeze target axes before each optimization run.
2. Freeze calibration/null procedures.
3. Separate development and holdout prompt populations.
4. Hide holdout outputs from the synthesizer.
5. Periodically test surviving funnels with an independent embedding instrument.
6. Retain axis-free displacement statistics.
7. Retain collateral dimensions.
8. Test paraphrase equivalence.
9. Test across independently sampled seeds.
10. Version every candidate and measurement artifact.
11. Never allow the optimizer to redefine its current judging axis.
12. Allow the falsifier to terminate the entire funnel hypothesis.

An optimized funnel that works only against one embedding instrument is an
instrument exploit until independently replicated.

---

## Initial experiment

The first implementation should deliberately avoid building the autonomous
optimizer.

Start by establishing whether the measurable object exists.

### Experiment 0A — Known funnel

Choose one previously observed compact directive with a strong qualitative effect.

Example class:

```text
“Respect the precise definitions when using terms of art.”
```

Construct:

- baseline condition;
- original funnel;
- 5–10 semantic paraphrases;
- length-matched neutral controls;
- semantically bleached controls.

Run a fixed prompt population across multiple seeds.

Store paired outputs.

Measure:

- displacement distribution;
- target projection;
- displacement coherence;
- residual displacement;
- paraphrase agreement.

Success criterion:

> The original and multiple paraphrases produce statistically separable,
> directionally consistent population effects relative to controls.

The draft protocol / preregistration scaffold for this experiment is
[`docs/experiments/2026-09-23-funnel-experiments-0a-0b-preregistration.md`](../../experiments/2026-09-23-funnel-experiments-0a-0b-preregistration.md)
under stable handle `SKM-FUNNEL-EXP0` / experiment handle `SKM-FUNNEL-EXP0A`. It does
not register an empirical run until a separate versioned run manifest freezes its unset
choices.

### Experiment 0B — Clause decomposition

If the funnel contains multiple clauses, test:

```text
F1
F2
F3
F1 + F2
F1 + F3
F2 + F3
F1 + F2 + F3
```

Calculate:

- constituent displacement vectors;
- Gram matrix;
- target projection;
- observed combination effect;
- additive prediction;
- residual interaction term.

This is the first direct test of reinforcing alignment.

The draft protocol / preregistration scaffold for this experiment is
[`docs/experiments/2026-09-23-funnel-experiments-0a-0b-preregistration.md`](../../experiments/2026-09-23-funnel-experiments-0a-0b-preregistration.md)
under stable handle `SKM-FUNNEL-EXP0` / experiment handle `SKM-FUNNEL-EXP0B`. It does
not register an empirical run until a separate versioned run manifest freezes its unset
choices.

### Experiment 0C — Recursive persistence

Select a controlled multi-turn task.

Insert the funnel once.

Measure the target displacement after each subsequent turn without reinserting it.

Then compare with:

- funnel reinserted every turn;
- no funnel;
- neutral directive.

Estimate decay/persistence before attempting a full transition operator.

A simple first observable is:

```text
a_t = projection(d_t, v_F)
```

plotted against turn depth.

Do not fit an eigenmode until the data warrants one.

---

## Minimal code-layer contract

The first implementation should require very little new machinery.

Suggested experimental record:

```json
{
  "experiment_id": "...",
  "model": "...",
  "instrument": "...",
  "prompt_id": "...",
  "seed": "...",
  "condition": "baseline|funnel|control|constituent",
  "funnel_id": "...",
  "constituents": ["F1", "F2"],
  "turn": 0,
  "response_ref": "...",
  "embedding_ref": "...",
  "provenance": {}
}
```

Derived sk-mcp artifacts should be reproducible from these records.

Initial operations needed:

```text
paired_displacements(...)
population_direction(...)
project_population(...)
displacement_gram(...)
compare_composition(...)
dispersion_change(...)
turnwise_projection(...)
```

These can begin as experiment-layer functions.

Do not prematurely promote them into stable MCP primitives.

---

## Architectural boundary

### sk-mcp owns

- embeddings;
- calibration;
- population geometry;
- paired displacement;
- null comparisons;
- projections;
- Gram/alignment measurement;
- dispersion measurement;
- provenance-bearing results.

### prompt-prix / pi owns

- corpus selection;
- candidate construction;
- model invocation;
- seed scheduling;
- search state;
- mutation/recombination;
- candidate versioning;
- holdout orchestration.

### falsifier owns

- independent replication;
- alternate instruments;
- null challenges;
- holdout evaluation;
- claims permitted by the evidence.

This ownership boundary is load-bearing: **sk-mcp measures; it does not optimize.**

---

## Rationale

Manual prompt engineering asks:

> What wording seems to work?

Funnel measurement asks:

> What compact contextual intervention reproducibly changes a response population?

Reinforcing alignment asks a stronger question:

> Are apparently different useful constraints independently pushing behavior in
> compatible geometric directions?

Autonomous synthesis then asks:

> Can a search process discover a small set of upstream constraints whose
> behavioral effects reinforce one another while unrelated effects remain small?

Recursive measurement asks the strongest current question:

> Does the resulting behavior feed context back into the system in a way that
> reinforces the same measured mode over subsequent turns?

These are experimentally separable questions.

That separation is the primary architectural value of this ADR.

### Positive consequences

- Gives “funnel” a falsifiable operational definition.
- Makes reinforcing alignment measurable rather than metaphorical.
- Reuses existing sk-mcp population geometry.
- Creates a concrete bridge between sk-mcp and prompt-prix/pi.
- Supports autonomous prompt discovery without an unconstrained LLM judge.
- Exposes additive, antagonistic, redundant, and superadditive directive structure.
- Makes persistence and recursive amplification measurable.
- Produces reusable datasets for later activation-space comparison.
- Allows strong claims to fail independently rather than collapsing into one
  funnel/no-funnel verdict.

### Negative consequences

- Output geometry may not correspond cleanly to internal representation geometry.
- Embedding anisotropy can create false apparent alignment.
- Pairwise cosine can conceal nonlinear structure.
- Optimization can Goodhart the embedding instrument.
- Large factorial clause experiments become expensive quickly.
- Recursive conversational effects can be confounded by ordinary topic drift.
- Apparent contraction can be generic homogenization rather than an attractor.
- Superadditivity depends on the metric and decomposition chosen.
- Cross-model transfer must be measured rather than assumed.

---

## Alternatives considered

### Manual prompt engineering

Useful for hypothesis generation.

Rejected as the measurement layer because memorable outputs do not establish
population geometry.

### LLM-as-judge optimization

Useful as a secondary qualitative instrument.

Rejected as the primary objective because the optimizer can learn judge
preferences without preserving interpretable behavioral geometry.

### Single scalar funnel score

Rejected.

It creates an obvious Goodhart surface and destroys diagnostically important
structure.

### Internal activation measurement first

Deferred.

Internal geometry would be highly valuable, but requiring activation access would
unnecessarily block the first behavioral experiment.

Emission-space measurement establishes whether there is a phenomenon worth
locating internally.

### Assume constituent directives are independent

Rejected.

Testing interaction among constituent effects is now one of the primary purposes
of the experiment.

---

## Open questions

- [ ] **Choose the first historically observed funnel.**
  **Resolution:** select a compact intervention with repeated qualitative evidence.

- [ ] **Select target behavioral axes.**
  **Resolution:** preregister from independently labeled positive/negative examples.

- [ ] **Decide which sk-mcp geometry is canonical for the first experiment.**
  **Resolution:** run existing valid centered/calibrated alternatives and report
  disagreement rather than selecting retrospectively.

- [ ] **Determine whether whitening materially changes constituent alignment.**
  **Resolution:** compare alignment matrices under registered geometry variants.

- [ ] **Define a robust interaction statistic for superadditivity.**
  **Resolution:** begin with target-axis interaction residuals before using norm
  comparisons.

- [ ] **Determine minimum sample count for stable Gram structure.**
  **Resolution:** bootstrap stability as N increases.

- [ ] **Determine whether attractor language is warranted.**
  **Resolution:** require both directional convergence and selective variance
  contraction across heterogeneous inputs.

- [ ] **Determine whether recursive `λ` estimation is stable enough to justify a
  transition model.**
  **Resolution:** first measure simple turnwise amplitude curves.

- [ ] **Test cross-instrument replication.**
  **Resolution:** repeat surviving results using at least one independent embedding
  model and direct behavioral labels.

- [ ] **Test cross-model transfer.**
  **Resolution:** only after establishing the phenomenon on one checkpoint.

---

## Claim discipline

Permitted initial claim:

> A contextual intervention produces a reproducible behavioral displacement in
> measured response space.

Stronger claims require additional evidence:

> Its constituent directives produce mutually aligned displacement fields.

> Their composition exhibits measurable positive interaction.

> The intervention creates attractor-like behavioral dynamics.

> Its effects recursively reinforce across conversational turns.

None of these imply, without separate internal measurement:

> The same vector or attractor literally exists inside the model.

The measurement ladder exists specifically to keep those claims separate.

---

## Supersession Relationships

**Supersedes:** — (additive; this ADR refines and adds to ADR-SKM-008 by defining
matched-population funnel measurement, constituent alignment, and composition; it does
not supersede ADR-SKM-008 or an existing tool).

**Superseded by:** TBD.

## Implementation Notes

The seven NumPy experiment-layer operations listed above and their contract tests land under
this ADR. Empirical 0A/0B runs, promotion of these operations to MCP primitives, autonomous
funnel synthesis, and optimizer work remain out of scope. Passing unit tests for the
experiment-layer operations is not funnel evidence; tests establish only that the numerical
implementation satisfies its code contract. Funnel evidence can come only from a separately
versioned, fully frozen empirical run, not from this implementation landing.
