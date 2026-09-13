# Trajectory analysis (position / rhythm regime)

This document describes the trajectory analyzer in
`semantic_kinematics/mcp/commands/trajectory.py` — what it measures, the physics
analogy behind it, the two detectors it ships (`deadpan_score`, `heller_score`),
and the limits of what it can see. It is the **position / rhythm** instrument;
its relationship to the separate **bearing / motion** regime is set out at the
end.

Formulas below were checked against the code. Where the prose and the code
diverge, the discrepancy is flagged inline rather than smoothed over.

## The core insight

Treat a passage as a particle moving through embedding space. Each sentence is a
point; the ordered sequence of points traces a path. The natural first guess is
that "weird" text — non-sequitur comedy, absurdist prose — should show high
**curvature**, sharp turns in the path. That guess is wrong.

In high-dimensional embedding space, curvature is uniformly high and therefore
uninformative. This holds at ~4096-d (NV-Embed-v2) and is still true at 768-d
(embeddinggemma). The angular curvature between consecutive displacement vectors
clusters near π/2 for random walks, for coherent prose, and for absurdist comedy
alike — nearly all pairs of high-dimensional vectors are close to orthogonal, so
the measurement saturates and discriminates nothing.

The signal that does carry information is **acceleration**: the rate at which
pacing changes. A passage can move quickly or slowly through semantic space; what
distinguishes structures is *how abruptly that speed changes*, not how sharply the
path turns.

## Physics analogy

| Quantity | Definition | Intuition |
|---|---|---|
| Position | `e[i]` | semantic content of sentence `i` |
| Displacement | `e[i+1] − e[i]` | the semantic shift from one sentence to the next |
| Velocity | `‖e[i+1] − e[i]‖` | pacing magnitude (how far the topic moved) |
| Acceleration | `\|v[i+1] − v[i]\|` | rate of pacing change |

Read concretely: a velocity of ~0.8 means a large topic shift between two
sentences; an acceleration of ~0.4 means the pacing itself changed sharply — the
text sped up or slowed down between one step and the next.

Note that velocity is a **norm** — it discards direction. This instrument is
about *cadence and magnitude*, not about which way the text moved. Direction is
the concern of the bearing regime (see below).

## Two species of absurdism

The analyzer ships two structure detectors, named after two distinct comedic
forms.

### Adams — the deadpan snap

Calm, calm, calm, SPIKE, calm. The Douglas-Adams pattern is an isolated
acceleration spike sitting on an otherwise stable background — a single sharp
swerve delivered without buildup. Detection rewards:

- **spikiness** — few spikes, not many (an everywhere-spiky passage is just
  noisy, not deadpan);
- **isolation** — each spike stands well above its immediate neighbors;
- **background stability** — the non-spike acceleration is low and steady;
- **contrast** — the largest spike dwarfs the loudest background step.

```
deadpan_score = 0.25·spikiness
              + 0.35·mean_isolation
              + 0.20·background_stability
              + 0.20·contrast
```

(Weights confirmed against `compute_deadpan_score`.)

### Heller — the bureaucratic trap

The Joseph-Heller pattern is circular: structures that decelerate into
tautological loops. Velocity decreases over the passage, later sentences
reference earlier ones, and the net displacement from start to end tends toward
zero — the text moves a lot but goes nowhere. Detection rewards:

- **circularity** — high similarity between `e[i]` and `e[i−2]`, the path folding
  back on itself;
- **tautology density** — high pairwise similarity combined with low net
  displacement relative to total path length;
- **deceleration** — a downward trend in velocity.

```
heller_score = 0.35·circularity
             + 0.40·tautology_density
             + 0.25·deceleration
```

(Weights confirmed against `compute_heller_score`.)

## Metrics computed

For a passage of `n` sentences the analyzer produces:

- a **velocity profile** `[v_0 … v_{n-2}]` (one per consecutive pair), and
- an **acceleration profile** `[a_0 … a_{n-3}]` (one per consecutive velocity
  pair).

  *Note on indexing:* the prompt brief describes these as `[v_0 … v_{n-1}]` and
  `[a_0 … a_{n-2}]`. In the code, `compute_velocities` returns `len(embeddings) −
  1` values and `compute_accelerations` returns `len(velocities) − 1`, so for `n`
  sentences there are `n − 1` velocities and `n − 2` accelerations. The lengths
  above reflect the code.

**Spike detection** (`detect_acceleration_spikes`). An acceleration `a_i` is
flagged as a spike when `a_i ≥ threshold` (default `0.3`). Each spike records:

- `index` — its position in the acceleration profile;
- `magnitude` — `a_i`;
- `isolation_score` — `tanh(2·(a_i − mean_neighbors) / (a_i + 0.01))`, clamped at
  0 below (negative isolation is floored to 0). `mean_neighbors` is the mean of
  the one or two adjacent acceleration values;
- `position_ratio` — the spike's fractional position.

  *Discrepancy:* the brief gives `position_ratio = i / (n − 1)` with `n` the
  sentence count. The code computes `i / max(len(accelerations) − 1, 1)` — i.e.
  it normalizes by the length of the **acceleration profile** (`n − 2`), not by
  the sentence count. This still yields a 0..1 ratio across the spike series, but
  the denominator is `n − 2`, not `n − 1`.

**Curvature.** `compute_curvatures` returns the angle (`arccos` of the cosine)
between consecutive displacement vectors. It is computed and reported
(`max_curvature`, `max_curvature_index`, optional `curvature_profile`) but, per
the core insight, is expected to saturate near π/2 and is not used in either
score.

**Circularity** (`compute_circularity`). Mean cosine similarity between `e[i]`
and `e[i−2]` across the passage, then rescaled `(mean_loop_sim − 0.3) / 0.5` and
clipped to `[0, 1]`. The brief states the bare similarity; the code applies the
linear rescale before use.

**Tautology density** (`compute_tautology_density`). Combines two parts:

- a similarity term — mean pairwise cosine over all sentence pairs, rescaled
  `(mean_sim − 0.3) / 0.5` and clipped;
- a displacement term — `1 − min(displacement_ratio, 1)` where
  `displacement_ratio = ‖e_end − e_start‖ / (total_path_length + 0.01)` and
  `total_path_length` is the sum of velocities.

  These are blended `0.6·similarity + 0.4·displacement`.

  *Discrepancy:* the brief describes "mean pairwise cosine + displacement ratio
  `‖e_end − e_start‖ / total_path_length`". The code uses **one minus** the
  displacement ratio (low net displacement → high score) and combines the two
  terms with fixed `0.6 / 0.4` weights rather than a plain sum. Both rescaling
  and the inversion matter for interpreting the value.

**Deceleration** (`compute_deceleration`). Blends the proportion of
velocity-decrease steps with a trend term from the linear-fit slope of the
velocity profile: `0.6·decel_ratio + 0.4·trend_score`, where `trend_score` is
`1.0` for slope `< −0.05`, `0.5` for slope `< 0`, else `0.0`.

**Comparison / fitness mode** (`compare`, used by `compare_trajectories`). For
scoring a synthetic passage against a golden reference, the analyzer combines
absolute-quality and relative-similarity terms into a single `fitness_score`
(lower = better match):

```
fitness_score = 0.30·(1 − synthetic_deadpan)
              + 0.15·(1 − synthetic_mean_isolation)
              + 0.20·min(acceleration_dtw, 1)
              + 0.15·(1 − max(acceleration_correlation, 0))
              + 0.10·(1 − spike_position_match)
              + 0.10·(1 − spike_count_match)
```

The rhythm term uses **Dynamic Time Warping** on the two acceleration profiles;
the remaining terms cover acceleration correlation, spike-position alignment, and
spike-count agreement.

## What this doesn't capture

- **Semantic content.** The instrument knows that a passage's *pacing* matches a
  pattern; it does not know *what* is funny, or what the text is about.
- **Lexical and register features.** Word choice, tone, irony markers, dialect —
  none of these are visible to a pacing measure.
- **Multi-paragraph structure.** It operates over a flat sentence sequence; it
  has no model of paragraph-, scene-, or section-level organization.

## Typical values

These are the instrument's **design-intent** values, not freshly measured
results. Read them together with the provenance caveat in the next section before
trusting any of them.

| Passage | deadpan | heller |
|---|---|---|
| Adams Vogon scene | ≈ 0.72 | — |
| Heller, *Catch-22* | — | ≈ 0.56 |
| Generic "wacky" | ≈ 0.70 | ≈ 0.30 |
| Dry prose | ≈ 0.58 | ≈ 0.24 |

Intended discriminator: **random weirdness scores high deadpan but low heller**;
**crafted absurdism shows both**. A passage that is merely odd produces isolated
acceleration jumps without the circular, decelerating loop structure; deliberately
constructed absurdism produces both at once.

## Provenance & non-reproduction caveat

The typical-values table above is the instrument's **design intent**, validated
historically on NV-Embed-v2 (4096-d) over **hand-cut** segments. It has **not**
been reproduced on the current embedder and atom.

The specific founding claim — "Adams Vogon scene = deadpan 0.72" — traces to a
qwen3.5-9b gist that **narrated** a jolt rather than measuring one. That gist
relabeled drift as acceleration, hand-cut its own segments, and never computed
the second derivative. It is not a measurement.

When the bypass-dialogue specimen is run on **embeddinggemma-300M (768-d)** at the
**sentence atom**, the trajectory reads **flat**: the maximum acceleration lands
on scene-change narration, not on the punchlines. The detector's premise — that
crafted comic timing produces isolated, isolated-and-contrasting acceleration
spikes — does not survive that change of embedder and atom.

Consequences:

- Do **not** re-trust the typical-values table unaudited. Treat those numbers as
  targets the instrument was built to hit on a different embedder, not as
  observed behavior of the current pipeline.
- This non-reproduction is the motivation for **ADR-SKM-0003
  (context-conditioned embedding atom)**: the sentence-as-point atom may be the
  wrong unit for the signal this instrument is trying to detect.

See `docs/ADRs/proposed/ADR-SKM-0003-context-conditioned-embedding-atom.md`.

## Relationship to the bearing regime

This trajectory instrument is the **position / rhythm** regime. There is a
separate **bearing / motion** regime that asks a different question with a
different atom and a different null.

| Regime | Question | Atom | Null | Tool |
|---|---|---|---|---|
| Position / rhythm | "where does this sit / how does it drift?" | sentence (point) | corpus-null (where real text sits) | ADR-SKM-0007 axis-alignment; this trajectory analyzer (velocity / accel / curvature) |
| Bearing / motion | "which way did this move vs a named axis?" | displacement (+ anchor pair) | measured-displacement-null (random motion under anisotropy) | ADR-SKM-0001 projection primitive |

The position regime is **magnitude and cadence only**. Velocity is a norm and so
discards direction; curvature and drift are reflexive angles between the path's
own segments, and those saturate near π/2 in high dimensions. None of these
quantities asks *which way* the text moved relative to a named axis — that
referential, direction-bearing question belongs to the bearing regime, where
high dimensionality sharpens the measurement rather than washing it out.

See:

- `docs/ADRs/proposed/ADR-SKM-0001-directional-projection-primitive.md`
- `docs/ADRs/proposed/ADR-SKM-0003-context-conditioned-embedding-atom.md`
