# Context-conditioned embedding atom for conditional-jolt detection

**Status**: `proposed`
**Date**: 2026-06-16
**Related**: ADR-SKM-0001 (directional projection primitive — this ADR supplies a second, independent motivation for it)
**Scope**: semantic-kinematics-mcp (sk-mcp)

---

## Context

The founding belief that the HHGG Vogon/absurdism passage "jolts" was traced (2026-06-15, §8) to a qwen3.5-9b gist that **narrated** a jolt rather than measuring one: it relabeled `drift` as `acceleration`, hand-cut its segments, ran nomic via a silent default (#14), and never computed the second derivative. The Stage-1 jolt smoke harness (`scripts/smoke_jolt.py`, a motivating fixture, **not** the instrument) then produced the clarifying result:

- **Specimen A** (Adams bypass dialogue, sentence atom, embeddinggemma) read FLAT. Max acceleration landed on the narration line "shadow over Arthur Dent's house," not on the punchlines ("With a torch.", "Beware of the Leopard"), which did not spike.
- **Specimen B** (escalation conversation, per-turn vault vectors) showed strong isolated spikes — but the max landed on the "O(2^n) complexity" **topic-shift** turn.

The between-specimen separation is confounded (A is bypass dialogue at sentence atom; B is escalation at turn atom — atom size and content type are free to do the work). The line that survives is **within-specimen**: raw acceleration fires reliably on scene-change and topic-shift, and never on the named comedic discontinuity. Magnitude finds *a* discontinuity; never *the* one.

This left a fork — is the jolt (a) sub-sentence granularity that sentence-mean low-passes, recoverable by a finer atom, or (b) bearing rather than magnitude at any scale? The first instinct (token-wise, then phrase-wise) treats it as a granularity problem. Working the geometry showed it is not only that.

## The reframe driving this decision

**The jolt is a conditional object.** It does not live in the punchline's standalone vector; it lives in the punchline-*as-read-after-the-setup*. The incongruity is bureaucratic calm against planetary demolition — and the demolition must be present in the representation when the calm lands. Terminal punctuation (the hard stop on "With a torch.") encodes the *delivery* but cannot encode the *catastrophe it underplays*. Therefore an embedder fed any segment in isolation — token, phrase, or sentence — amputates the contrast that constitutes the jolt. Finer context-free segmentation does not recover the jolt; it differences a sequence of contextless points, and the "acceleration" it reads is topic/lexical change, i.e. exactly the scene-shift and O(2^n) spikes already observed.

This also separates two contexts that were being conflated: the **embedder's input window** (how much text is encoded into one vector) and the **analyzer's difference window** (how many vectors the second derivative spans). "How far back" is a question about the former.

## Decision

Adopt a **context-conditioned embedding atom** for jolt/register detection:

1. **Atom is conditioned, not free.** Embed a target span inside a `[leading context + target]` window so attention encodes the contrast into the target's representation. Do not embed the target in isolation.
2. **Pool over the target span only**, not the whole window. Whole-window mean pooling dilutes the target back toward the cone center (the low-pass problem returns through the back door); span-restricted / end-weighted pooling keeps the target's representation while letting attention see the setup.
3. **Leading-context length is determined empirically by a context-ramp**, not fixed a priori. Embed the target with `k` leading atoms for `k = 0, 1, 2, …` and read where the representation lands as a function of `k`. If the jolt is conditional there is a critical `k` below which it is invisible and above which it appears; that critical `k` *is* the setup length, read off directly.
4. **Segment on punctuation, not paragraph ends.** Phrase-scale units (split on sentence-internal and terminal punctuation) give punchline resolution. Paragraph boundaries are scene/topic shifts — the discontinuity magnitude already fires on — so segmenting there reconcentrates exactly what we are trying to see past. Include the **trailing demarcator** token in the target span (the period/dash/ellipsis/bang is part of the register-act).
5. **Measured-displacement null first, per-`k`, length-stratified.** Compute the null at each granularity/`k` and read it **before** looking at any acceleration or projection number. This is non-negotiable: reading raw kinematics against sentence-scale priors is the nomic-silent-default failure in a finer mesh.
6. **Seam.** Phrase-pooled vectors feed `TrajectoryAnalyzer.analyze_embeddings(matrix)` (landed in PR #26, `a2ee10e`) with no further plumbing.

## Rationale

**Why conditioning is mandatory.** The jolt is the target read after its setup; context-free embedding at any granularity removes the contrast, so the only thing left to measure is topic/lexical change — which is what the flat/scene-shift readings already are. Conditioning is the precondition for the jolt being present in the data at all.

**Why this favors bearing over magnitude — on geometric grounds, not empirical luck.** If adjacent target vectors are embedded with overlapping left-context, consecutive vectors *share input* and are therefore artificially close, which **smooths the second difference** (acceleration). The very construction that makes the jolt encodable suppresses the jolt's magnitude signature. Projection onto a situatedness axis reads the target vector's **absolute location** (did it land in deadpan-over-catastrophe territory) and is immune to how much its neighbor's window overlapped. So adding the setup is mandatory to see the jolt and simultaneously kills the magnitude read while sparing the projection. This is a second, independent motivation for the ADR-SKM-0001 directional-projection primitive, arriving from the embedding side rather than from anisotropy.

**Why the null is length/`k`-stratified.** Short pooled segments retain more idiosyncratic, off-cone direction variance than long ones, which have averaged back toward the cone center; unit-norming removes length-as-magnitude but not length-as-direction-variance. Punchlines are short, so raw phrase acceleration will spike on them for a length reason that mimics the comedic reason — and would feel like confirmation. The context-ramp adds a second entanglement: longer windows (larger `k`) are more cone-centered, so `k` and pooling variance are confounded. Both must be absorbed by the null before any number is read.

### Positive consequences
- The jolt becomes encodable at all; context-free atoms never could.
- One preprocessing substrate feeds both the magnitude run (#25 item 1) and the bearing primitive (#25 item 4).
- The context-ramp is a real instrument: it measures setup length rather than guessing it.
- Strengthens ADR-SKM-0001 with a motivation independent of the anisotropy argument.

### Negative consequences
- **Breaks the magnitude channel for this task** via overlap-smoothing. Accepted, because bearing is the target — but it means magnitude cannot serve as the validation here.
- Adds `k` as a confound entangled with pooling variance/length; more null bookkeeping.
- The pooling choice is gated on the embedder's architecture (see Open Questions) — an external dependency not yet confirmed.
- Demarcator-*type* becomes a live trajectory coordinate; spikes tracking punctuation-type changes are a new artifact to rule out (the function-word problem rescaled to boundaries).

## Alternatives Considered

### Option A: Token-wise, context-free
**Rejected.** Token representations are *more* cone-collapsed than pooled spans, so anisotropy worsens at token granularity. Worse, token-wise acceleration is dominated by the function-word/content-word alternation at every step — the kinematics of "the/of/a" would drown a four-token punchline. Finer mesh, more noise, same blindness to the jolt.

### Option B: Phrase atom, context-free (split on punctuation, embed in isolation)
**Rejected as insufficient, not wrong.** It captures the target's own delivery shape (the hard stop = deadpan) but amputates the setup, so it cannot encode the incongruity. Differencing contextless phrase points reads topic/lexical change — the scene-shift spikes already seen. It survives only as the *pooling span* inside the conditioned window (Decision 2), not as the embedding unit.

### Option C: Sentence-mean (status quo)
**Rejected.** This is the founding instrument's granularity. It low-passes the jolt by averaging it into surrounding narration — the original FLAT reading on Specimen A.

### Option D: Whole-window mean pool over `[setup + target]`
**Rejected.** Every leading token contributes equally, diluting the target back toward the cone center as `k` grows. The low-pass problem returns through the back door. This is why pooling is span-restricted/end-weighted (Decision 2), conditional on the embedder supporting it.

## Open Questions

- [x] **Is embeddinggemma (:8082, 768d) bidirectional, and what is its pooling — mean vs. last-token/end-weighted?** **RESOLVED 2026-06-16 (measured against live :8082, embeddinggemma-300M-F32, n_ctx 2048):** (a) Default pooling is **mean** (`--embeddings`, no `--pooling`); the OpenAI-compat `/v1/embeddings` path always returns one pooled vector. (b) **Per-token output is available** by launching with `--pooling none`; the native `/embeddings` endpoint then returns `(n_tokens, 768)` (confirmed `(7,768)` for a 7-token input). Per-request `pooling` override is ignored — it is a launch flag. (c) The model is **bidirectional**: appending tokens shifts earlier shared-prefix token reps (`"the"` cos 0.79 between `"the cat"` and `"the cat sat on the mat"`; causal would be 1.000). **Branch picked:** feed `[setup + target]` as one window, pool the target span client-side off `/embeddings` under `--pooling none`. **Caveat (off-distribution):** embeddinggemma is calibrated for mean-pool over the *whole* prompted input (with task prefixes); the per-token reps are properly contextualized, but pooling a target *sub-span* is not how the sentence representation was trained — raises the prior that the per-`k` length-stratified null absorbs more apparent signal than expected. **Because the ramp window is jointly (not causally) encoded, `k` measures joint-window size, not strictly "how far back."**
- [ ] **What is the critical `k` (setup length) for the Vogon specimen, and is it constant or contrast-dependent?** **Resolution:** Context-ramp experiment, #25 item 1.
- [ ] **Does the overlap-smooths-acceleration prediction hold empirically — magnitude degrading as left-context overlap grows while projection is preserved?** **Resolution:** Measure both channels across the ramp; this is the decisive bearing-over-magnitude test from the embedding side.
- [ ] **Does the per-`k`, length-stratified null behave as expected (longer windows → more cone-centered)?** **Resolution:** Compute and inspect the null before any kinematic read; ties to the ADR-SKM-0001 primitive build (#25 item 4).
- [ ] **Do demarcator-type changes (period/dash/ellipsis/bang) produce spurious spikes?** **Resolution:** Confirm spikes do not track punctuation-type transitions when validating the phrase atom.

**Discipline:** the token-wise/phrase-wise run must be built to **falsify** "a [token|phrase]-scale jolt exists," not to hunt for one. If the punchlines stay flat even isolated at their own scale, length-corrected and context-conditioned, that is the cleanest case yet for bearing-not-magnitude and aims the run straight at the situatedness anchor that does not yet exist.

### Verdict pre-registration (precondition — not an open question)
The verdict only means something if its terms are fixed **before any embedding**. Author and commit a `specimen_registration` recording, from the segmented text alone: (a) the **punchline step indices**; (b) the **control indices** (known scene-shift over-firers, equal-formatting); (c) the **flag sigma** `S`; (d) the **decision rule** — POSITIVE iff a punchline clears `S` in its `(k, length, demarcator)` stratum at a `k*` **and stays cleared contiguously across the usable-`k` band above `k*`**, while at those `k` (i) equal-stratum controls do **not** clear and (ii) the punchline's **immediate neighbors** do **not** clear (local calm-SPIKE-calm isolation — the deadpan *shape*, not mere spike-existence); NEGATIVE iff no punchline meets this under `S`. Testing only named indices bounds the comparison count (multiplicity controlled by construction). **Contiguity is the falsifiable shape:** once the setup enters the window (`k ≥ setup-length`) the jolt cannot be un-conditioned by more context and the per-`k` null has cancelled the smoothing, so it must persist — an isolated single-`k` crossing is per-cell stratum noise and is rejected.

### Confounds that gate the verdict (read discipline)
1. **`k`-artifact (defines the verdict).** deadpan↓ / heller↑ drift monotonically with `k` as pure geometry (shared leading context → adjacent target vectors pulled together → lower acceleration). The per-`k` null cancels this **only because** the null builder and the look call the **same** conditioned construction (smoothing in numerator and denominator). The verdict is the **null-calibrated flag pattern**; the raw deadpan/heller-vs-`k` curves are diagnostic-only.
2. **Stratification absorbs formatting, not content.** `TERM_ISOLATED` holds both the deadpan one-liner and the over-firing scene-change one-liner, so a null result means "magnitude can't separate comedic from scene discontinuity *at equal formatting*," NOT "no signal." A qualifying spike is a **new** finding (correctly-atomed instrument detects the deadpan shape), never vindication of the falsified NV-Embed/4096-d hand-cut gist, which stays falsified-as-measured. Verdict-stratum `n` must be checked at the punchline's own cell; if it backs off below the demarcator level, the claim drops to this weaker bound.
5. **`k=0` hard gate.** `k=0` is the context-free phrase (Option B) → must show context-free behavior (scene-shift-locked, *no isolated punchline spikes*); punchline spikes at `k=0` ⇒ atom or span-alignment broken ⇒ every `k>0` number suspect.

## Result (2026-06-16 — falsify-first run, pre-registered)

**VERDICT: NEGATIVE**, by the pre-registered rule (`data/registrations/bypass_dialogue_registration.json`; result `output/ramp_deadpan_bypass_result.json`).

- Specimen: HHGG bypass dialogue, 91 phrases. Detector: conditioned-phrase displacement magnitude vs the measured per-(k×length×demarcator) null (`data/nulls/conditioned_phrase_displacement_embeddinggemma768.json`, 280 turns; verdict cell `k|4-7|SET_QUOTE` n=1368, finest level — no backoff).
- Punchlines 72 (`'With a torch.'`) and 74 (`'So had the stairs.'`) never clear σ=3 at any k. Equal-format controls and the scene-shift-narration controls also stay within ~1σ. **Bound (confound 2): magnitude cannot separate comedic from scene/ordinary discontinuity *at equal formatting* — NOT "no signal."**
- The conditioned construction is sound (live-validated span localization; per-`k` null cancels the smoothing artifact; raw deadpan 0.69→0.69 *declined* with `k`, the confound-1 artifact, diagnostic-only — not a detection). So the negative is about the **channel**, not the instrument.
- **Sub-finding (suggestive, not significant):** punchline 72's z rises monotonically with `k` (−1.66 → +0.82, ~2.5σ in the *predicted* direction as setup enters the window), but stays sub-threshold; 74 shows no such trend. Hints the conditional contrast exists as *direction-of-motion*, not excess *magnitude*.
- **Consequence:** the cleanest case yet for **bearing-not-magnitude**. The deadpan jolt is not a displacement-magnitude phenomenon at this atom/embedder even when conditioned → aims at the **projection / situatedness** track (ADR-SKM-0001), whose anchor does not yet exist. The NV-Embed/4096-d "deadpan 0.72" founding belief remains falsified-as-measured.

This resolves OQ "does the per-`k` null behave as expected" (yes — monotonic cone-centering, per-k overlap 0→140 chars) and largely answers "overlap-smooths-magnitude" (raw deadpan declines with `k`; the null absorbs it). The setup-length / critical-`k` OQ is moot for magnitude (no clearance to locate a `k*`).

## Supersession Relationships

**Supersedes:** none.
**Reinforces:** ADR-SKM-0001 (adds the embedding-overlap motivation for directional projection, independent of anisotropy). ADR-SKM-0001 should add a back-reference noting this second motivation.
**Superseded by:** TBD (a confirmed situatedness-anchor construction may absorb this once the projection primitive is built).

## Implementation Notes

| File | Change Type | Description |
|------|-------------|-------------|
| `src/.../trajectory_analyzer.py` | Existing seam | `analyze_embeddings(matrix)` / `analyze_segments(list)` (PR #26, `a2ee10e`) — context-conditioned phrase-pooled vectors plug in here as `matrix`. |
| `scripts/smoke_jolt.py` | Motivating fixture | NOT the instrument; the Stage-1 smoke that produced the flat/scene-shift readings. |
| semantic-chunker preprocessing | New | Punctuation-based phrase segmentation with trailing demarcator; context-ramp window construction. Note: the 2-exchange chunker win windowed *up* (more context per unit); phrase atoms window *down*, so that win is a warning here, not precedent. |
| `output/vectors/chunks.jsonl`, `embed_checkpoint.jsonl` | Existing vault | 80,520 per-turn, embeddinggemma 768d, unit-norm. Re-embedding at phrase granularity with conditioning is a vault-cost this ADR's atom change implies. |

**Specimens:** escalation `e7c2fe94…`; Vogon `0029_Absurdist_LLM_ideas`.

## References

- Provenance of the narrated jolt: qwen3.5-9b gist — https://gist.github.com/shanevcantwell/6c0344db773e11fce23591967f2e4572 (narrated a jolt; relabeled `drift` as `acceleration`; never ran the second derivative).
- §8 rev 2026-06-15 (Stage-1 jolt smoke + seam; nomic fossil; vault confirmed).
- ADR-SKM-0001 (directional projection primitive).
- Issue #25 (fresh-stab roadmap), #14 (nomic silent-default hard-fail), #28 (vault embed prereq), PR #26.
