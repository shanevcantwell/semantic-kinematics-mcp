# Axis Alignment — Function-Level Math Reference

This document specifies exactly what each function in
`semantic_kinematics/mcp/commands/axis_alignment.py` computes. It is the
mechanical companion to [ADR-SKM-0007](ADRs/proposed/ADR-SKM-0007-referential-axis-alignment.md),
which records *why* the design is shaped this way.

## Notation

- `d` — embedding dimensionality (4096 for NV-Embed-v2).
- Embeddings from the production backends are **L2-normalized** (unit vectors),
  so any dot product with a unit axis lies in `[-1, 1]`. The math below does not
  *require* normalization, but the tight null concentration that makes the
  z-score sharp depends on it.
- `eᵢ` — the embedding of sentence `i` in a passage of `n` sentences.
- `v̂_ref` — the unit reference axis (the semantic direction being measured).
- A "projection" of a vector `x` onto `v̂_ref` is the scalar dot product
  `x · v̂_ref` — signed distance along the axis.

All internal arithmetic is done in `float64` regardless of the embedding dtype,
to keep means and standard deviations stable.

---

## `_split_exemplars(raw: str) -> list[str]`

Splits an anchor string into exemplars on newlines, strips each line, and drops
empties.

- Input `"escalating threat\n urgent deadline \n\n"` → `["escalating threat", "urgent deadline"]`.

Multiple exemplars per pole are the recommended usage: a single phrase produces a
noisy axis, whereas averaging several exemplars (next function) yields a stable
direction.

---

## `build_axis(pos_vecs, neg_pole) -> (unit_axis, pole_separation)`

Constructs the reference axis from the positive exemplars and a negative pole.

**Inputs**
- `pos_vecs` — shape `(n_pos, d)`, the embedded positive-anchor exemplars.
- `neg_pole` — shape `(d,)`, the negative pole (either averaged negative
  exemplars, or the background-null mean — see `alignment_core`).

**Computation**

1. Average the positive exemplars into a single positive pole:

   `pos_mean = (1/n_pos) · Σⱼ pos_vecs[j]`

2. Form the raw axis as the difference of poles:

   `raw = pos_mean − neg_pole`

3. Measure pole separation as its length:

   `pole_separation = ‖raw‖₂`

4. Normalize to a unit axis:

   `v̂_ref = raw / pole_separation`

**Edge case.** If `pole_separation == 0` (poles coincide exactly), it returns a
**zero vector** and `0.0` rather than dividing by zero. Callers gate on the
separation before using the axis, so a zero axis is never projected in practice;
this is purely a NaN guard.

**Why a difference vector.** The axis points *from* the negative concept *toward*
the positive concept. Projection onto it answers "how far toward the positive
pole, away from the negative one, is this point?" — a signed, directional
quantity, unlike cosine similarity to a single anchor.

---

## `null_stats(null_embeddings, unit_axis) -> (mu0, sigma0)`

Computes the background null distribution **for this specific axis**.

**Computation**

1. Project every background embedding onto the axis:

   `projₖ = null_embeddings[k] · v̂_ref`   for each row `k`

2. Return the mean and population standard deviation of those projections:

   `μ₀ = mean(proj)`,  `σ₀ = std(proj)`   (NumPy default `ddof=0`)

`μ₀` captures where the anisotropic embedding "cone" sits along this axis — the
baseline offset that a raw dot product would mistake for signal. `σ₀` is the
natural spread of generic text along the axis, i.e. the yardstick for "unusual."

Because the axis is built from caller-supplied anchors at call time, this null
**cannot be precomputed** — it is recomputed per call from the cached raw
background embeddings (a single matrix–vector product, cheap even at tens of
thousands of rows).

---

## `alignment_core(sentence_embeddings, pos_vecs, neg_vecs, null_embeddings, min_pole_separation) -> dict`

The full instrument. Pure function — no IO, no spaCy, no embedding backend — so
it is exhaustively unit-testable with hand-built vectors.

**Inputs**
- `sentence_embeddings` — `(n, d)`, the passage.
- `pos_vecs` — `(n_pos, d)`, positive exemplars.
- `neg_vecs` — `(n_neg, d)` **or** `None`.
- `null_embeddings` — `(N, d)`, the background corpus.
- `min_pole_separation` — axis-quality floor (default `0.05`).

**Preconditions (each returns an `{"error": ...}` dict)**
- Fewer than 2 sentences → cannot measure a march.
- Fewer than 2 null embeddings → cannot form a distribution.
- `pole_separation < min_pole_separation` → `"axis underdetermined"`.
- `σ₀ == 0` → null has no spread along this axis; z-score undefined.

**Computation**

1. **Negative pole selection.** If `neg_vecs` is `None`, use the background mean
   as the negative pole:

   `neg_pole = mean(null_embeddings)`   (else `mean(neg_vecs)`)

   This is the de-meaning trick: with no explicit negative anchor, the axis runs
   from the cone center toward the positive concept, so the anisotropy offset is
   subtracted out by construction.

2. **Axis + gate.** `v̂_ref, pole_separation = build_axis(pos_vecs, neg_pole)`;
   reject if below the floor.

3. **Null calibration.** `μ₀, σ₀ = null_stats(null_embeddings, v̂_ref)`.

4. **Position projection.** Project each sentence onto the axis:

   `projᵢ = eᵢ · v̂_ref`

5. **Position trace (z-scored).** Standardize against the null:

   `zᵢ = (projᵢ − μ₀) / σ₀`

   Each `zᵢ` is "how many sigma along the axis, relative to background text,
   sentence `i` sits." This is the headline output, `position_zscores`.

6. **Step projections.** The per-step axis displacement is the first difference
   of the projection trace — and, by telescoping, exactly the projection of the
   inter-sentence displacement vector:

   `sᵢ = projᵢ₊₁ − projᵢ = (eᵢ₊₁ − eᵢ) · v̂_ref`

7. **Net and total step.**

   `net_step = Σ sᵢ = proj_last − proj_first`
   `total_step = Σ |sᵢ|`

8. **Axis-restricted straightness.**

   `axis_straightness = |net_step| / total_step`   (`0.0` if `total_step == 0`)

   This is the 1-D analog of the trajectory module's displacement ratio,
   **restricted to the axis**. `1.0` = every step moved the same direction along
   the axis (a disciplined straight-line march); `0.0` = motion along the axis
   canceled out (oscillation, no net progress). It is independent of whether the
   passage is "straight" in the full embedding space — only axis-projected motion
   counts.

9. **Axis drift.**

   `axis_drift = z_last − z_first = net_step / σ₀`

   The net signed march across the whole passage, in sigma units. Positive =
   moved toward the positive pole; negative = toward the negative pole.

**Returns** (rounded to 4 decimals)

| Key | Formula | Meaning |
|-----|---------|---------|
| `n_sentences` | `n` | Sentence count |
| `position_zscores` | `zᵢ` | Per-sentence axis position, sigma units |
| `axis_drift` | `z_last − z_first` | Net signed march |
| `axis_straightness` | `\|Σsᵢ\| / Σ\|sᵢ\|` | March discipline along the axis |
| `mean_zscore` | `mean(zᵢ)` | Average axis position |
| `pole_separation` | `‖pos_mean − neg_pole‖` | Axis quality |
| `null_count` | `N` | Background size behind the calibration |

**Reading the two motion numbers together.** `axis_drift` and
`axis_straightness` answer different questions and should be read jointly:

- High drift **and** high straightness → a sustained, disciplined march along the
  axis (the escalation signature).
- High drift **but** low straightness → it got there, but wandered.
- Low drift **and** high straightness → it held a steady position on the axis.
- Low drift **and** low straightness → axis-orthogonal motion; the passage moves,
  but not along *this* direction.

---

## `build_null_cache(adapter, texts, out_npy, source="") -> manifest`

Embeds a background corpus once and persists it for reuse. This is the heavy,
one-time step (it runs the embedding model over the whole corpus).

**Computation / side effects**
1. Normalize the output path to end in `.npy` (so the manifest's stored filename
   stays in sync with what `np.save` actually writes).
2. `embeddings = adapter.embed_batch(texts)` → shape `(N, d)`.
3. Save embeddings to `out_npy` via `np.save`.
4. Write a sibling manifest `out_npy + ".json"`:

   ```json
   {
     "model_name": "<adapter.model_name>",
     "dimensions": d,
     "count": N,
     "embeddings_path": "<basename of out_npy>",
     "source": "<provenance string>"
   }
   ```

The `model_name` key is load-bearing: it is how the handler refuses a null built
under a different backend's geometry.

---

## `load_null_cache(manifest_path) -> (embeddings, manifest)`

Inverse of the builder.

1. Read the manifest JSON.
2. Resolve `embeddings_path`: used as-is if absolute, otherwise joined to the
   manifest's directory (so the cache directory is relocatable).
3. `np.load` the embeddings.
4. Return `(embeddings, manifest)`.

Raises on missing file, missing key, or unreadable array; the handler converts
these into a clean error dict.

---

## `analyze_axis_alignment(manager, args) -> dict`  (MCP handler)

The async tool entry point. It does only IO and validation; all numerics
delegate to `alignment_core`.

**Sequence**
1. Read args: `text`, `anchor_positive` (required), `anchor_negative`
   (optional), `background_ref` (falls back to env `AXIS_NULL_MANIFEST`),
   `min_pole_separation`, `include_sentences`.
2. Split positive exemplars; error if none.
3. Error if no `background_ref` — z-scores are meaningless without a null, so
   there is **no silent fallback**.
4. `load_null_cache(background_ref)`; on failure return a clean error.
5. **Model-geometry guard:** if `manifest["model_name"] != adapter.model_name`,
   refuse — a null from a different backend lives in a different space and its
   sigmas would be nonsense.
6. Tokenize the passage with `TrajectoryAnalyzer.tokenize_sentences` (spaCy,
   shared with the trajectory module); error if fewer than 2 sentences.
7. Embed: passage via `analyzer.embed_sentences` (which calls
   `adapter.embed_batch`), positive exemplars, and negative exemplars if given.
8. Call `alignment_core(...)`.
9. On success, attach `model_name` and `background_ref` for provenance, and echo
   `sentences` if `include_sentences` is set.

---

## Worked micro-example

A 4-D space with the axis along the first coordinate; a symmetric null centered
at the origin (`μ₀ = 0` along the axis, `σ₀ = √0.5 ≈ 0.707`):

```
null   = [[1,0,0,0], [-1,0,0,0], [0,1,0,0], [0,-1,0,0]]
pos    = [[1,0,0,0]]          # positive pole on +axis
neg    = None                 # → negative pole = null mean = origin
axis   = [1,0,0,0] (unit), pole_separation = 1.0

sentences        proj    z = proj/0.707
[0,1,0,0]   →     0.0      0.00
[0.5,0.5,0,0] →   0.5      0.71
[1,0,0,0]   →     1.0      1.41

steps        = [0.5, 0.5]
net_step     = 1.0,  total_step = 1.0
axis_straightness = 1.0          # disciplined straight march
axis_drift   = 1.41 − 0.00 = 1.41 sigma
```

A passage moving only in coordinates 2–4 (orthogonal to the axis) would have
constant `proj`, giving `axis_drift = 0` and `axis_straightness = 0`: it moves,
but not along this direction. These two cases are exactly the
`test_aligned_march_*` and `test_orthogonal_march_*` cases in
`tests/test_axis_alignment.py`.
