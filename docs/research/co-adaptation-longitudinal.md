# Co-adaptation — the instrument turned on the operator

**Status:** incubating VISION / research-direction. Speculative, single-subject. Authored
2026-07-12 from a live working session. Not a founding aim (see [`SPINE.md`](../SPINE.md)); a
constellation cluster (see [`map.md`](../map.md)). Promotion/distillation per ADR-CON-0001.

---

## Thesis

One instrument, pointed at the **operator** instead of the model. SPINE Aim 3 measures how the
*model's* register drifted across generations. This measures how the *operator* co-adapted — the
mirror. Operator and model are two coupled systems leaving traces in one ~2.5-year corpus; the
instrument reads the **coupling**, not either party alone.

## Why single-subject is legitimate here

- **A rare natural experiment.** One stable subject, ~2.5 years, *multiple model generations* — so
  the forcing function (the model's register) is a **moving target**, not a constant — plus
  speaker-labeled turns. Self-baseline means the instrument cone *and* the operator's voice are
  both common-mode within the stream and **cancel** (the cone-null we agonize over elsewhere is
  free here — cf. the map's "Longitudinal per-user abrupt-change" node).
- **On the program filter:** cheap · mathematical · contrastive · falsifiable.
- **No consent gate.** The operator *is* the subject — unlike the population-scale longitudinal
  node, there is no third-party governance question.

---

## Node A — LLM-accent / reverse entrainment

Human linguistic accommodation *to the model* — register adoption. The "LLM accent."

**H0 (pre-registered, before any real run).** The operator's register in era *E* is closer to the
register of *the model of era E* than to a fixed baseline; and — because entrainment **lags** and
**overcorrects** — the response to a register *changepoint* (model release, system-prompt swap, RBR
retune) is a **lagged step-response with overshoot**, not an instantaneous echo. Passive independent
drift predicts *no overshoot*; overshoot is the discriminating signature.

**The measurement atom is not a 2-message pair.** The signal lives in the transient *around a
changepoint*, over weeks — so the analysis is anchored to changepoint dates and reads
lag / overshoot / settle, never turn-adjacency (which is dominated by topic coherence, not accent).

**Refinement — the tell is migration, not rate.** Early affect is *context-dependent* (strategic,
opener-positioned, predictable from the prior model turn); late affect is *context-independent*
(ubiquitous, decoupled). The sharp measure is the **entropy / context-dependence of affect
placement over time** — high marginal information early, collapsing toward zero late — weighted to
the opening position that was ritualized. Rate-increase is only the shadow of this collapse.

**Mechanism — phenomenological anchor (operator, 2026-07-12, rare first-person data).** Affect
openers ("Good!", "Well done!") began as *deliberate soft-prompt steering* of the forward pass — a
mechanistically sound intuition (mood/presuppositional priming shifts the sampled distribution).
Over ~2 years they **saturated**: a token present in *every* prompt is common-mode, carries no
marginal information (maximum frequency = minimum information), and stops steering. It migrated out
of the operator's *variance* and into the operator's *mean* (μ) — the vector nullification
subtracts. Subjectively: a functionless-but-compulsory ritual ("a turn signal at 3am on an empty
rural road"), physically wrong to omit. **Anisotropy from the inside** — a feature absorbing into
the cone of one's own voice.

**Pilot evidence — motivating, NOT confirming.**
`../../../semantic-chunker/data/tat_metrics/exclamation_analysis_2023-2025.json` (Bard→Gemini,
2023-10-30 → 2025-10-20; 10,397 exchanges; timestamps 99.5% real). Amplification 1.77× (AI:user
per-turn exclamation rate); opening-sentence ratio 4.76× — the "enthusiasm radiation" forcing
function quantified. Crude decile (exchange-order) peek: user/AI exclamation ratio climbs from
~0.06 early to **1.5–3.4 late** — the user *out-exclaims the receding AI* — consistent with delayed
adoption that persists past the source's decline (the predicted overshoot). But it is exchange-order
binned, confounded, and not changepoint-aligned: it *earns* the proper run, it is not itself evidence.

**First cut.** Monthly-resampled, release-aligned lag/overshoot on the timestamped file;
conversational channels only; tool-stub boilerplate filtered.

---

## Node B — Ideation-variability trajectory

**Reported signal (memory-sourced, ~2025; encoder UNRECOVERED).** A prior analysis surfaced a drop
in the operator's ideation/communication **variability** over ~2 years. The encoder used is not
currently recoverable from the tree (`scripts/forensics/structural_fingerprint.py` in
semantic-chunker is a candidate lead). Recorded here as **reported-not-grounded** — *step zero is to
recover the original artifact and its encoder*, so the signal is grounded rather than remembered.

**The measurable.** Dispersion trajectory of operator-turn embeddings over calendar time (variance /
participation-ratio / entropy of the operator's *own* distribution, binned by time).

**Signal ≠ interpretation — firewalled.** A model (Gemini or an early Sonnet) interpreted the drop
as evidence that AI use *reduced ADHD influence* on focus of ideation. Recorded for provenance, but
held as a **high-inference clinical claim the instrument cannot support and does not adjudicate** —
and flagged as itself a specimen of the completion/narrative bias the program studies (a model
manufacturing a tidy, flattering causal story from a noisy signal — kudzu-class). The instrument
measures **dispersion**; it does not measure cognition, diagnosis, or valence.

**Primary confound (from the 2026-07-12 finding) — channel composition.** The corpus's back half is
`claude_code`-heavy, and `claude_code` turns are ultra-low-variance agentic boilerplate (PC1 of the
nv4096 corpus is dominated by near-identical `[Tool use: Read]` stubs — see
`scripts/probe_axis_poles.py`). A raw "variability dropped" is at high risk of being "the corpus
shifted toward a low-variance channel," **not** "the operator's ideation narrowed." Channel
stratification + tool-stub filtering is a **hard precondition**, not a nicety.

**Dual valence — the instrument stays mute on it.** Even a channel-clean dispersion drop admits
opposite readings: *gain* (focus/coherence improved) vs *harm* (externally-imposed narrowing /
cognitive capture — the operator's own Aim-3 "the mirror re-grinds me each release" cost). The
instrument reports the geometry; valence is out of scope.

**First cut.** Channel-stratified operator-turn dispersion over calendar time, fixed encoder,
against a null — *after* step zero (recover the artifact/encoder).

---

## The unifying hypothesis (why A and B may be one thing)

Node B may be a **consequence** of Node A: entraining to a bounded external register mechanically
narrows one's own dispersion. If the operator adopts the model's more-regular register, operator-turn
variance drops as a *side effect of accent adoption* — no cognitive story required. This links the
nodes and is a third firewall on the ADHD reading. **Testable:** does the dispersion drop concentrate
along the same axes the accent moves along? If yes, B is A's shadow.

---

## Shared rigor requirements

- **Fixed, recorded encoder** — ground the signal, don't remember it (Node B step zero).
- **Channel stratification** (conversational vs agentic); tool-stub filtering.
- **Changepoint anchoring** — release / system-prompt / RBR dates as annotated steps.
- **Contrastive / within-channel**; lag & overshoot *dynamics*, never a 2-message atom.
- **A null and a noise floor**; a claim that can be wrong in a checkable way.
- **H0 stated before the run**; a clean NEGATIVE is a gain held (a path closed).

## Epistemic & ethical stance

Measure geometry; do not narrate cognition. The subject may report phenomenology (rare and valuable
— see Node A's mechanism anchor); the instrument reports dispersion and drift. **Neither licenses a
clinical or identity claim.** The clinical interpretation is the single sharpest place the
completion-bias would manufacture a flattering story — held at arm's length by design. This whole
cluster is the founding concern (SPINE roots, turn 8913 — *"weapons grade… neurotransmitter-enhanced…
propaganda and sloganeering tools"*) arriving at its own doorstep: the one specimen who can be
*measured* and also *asked what it felt like*.

## Provenance & pointers

- **Session:** 2026-07-12 working session — nv4096 cone PCA + top-axis pole probe + these nodes.
- **Instrument (this session):** `scripts/measure_cone.py`, `scripts/probe_axis_poles.py`.
- **Pilot:** `semantic-chunker/data/tat_metrics/exclamation_analysis_2023-2025.json`.
- **Corpus:** tvi nv4096 (`chunk_id`+`embedding`) ⋈ `output/vectors/chunks.jsonl`
  (`speaker`/`timestamp`/`source`/`text`).
- **Family:** SPINE Aim 3 (mirror); `map.md` "Longitudinal per-user abrupt-change" (same subject,
  self-baseline); Aim 2 / kudzu (the model's *moves* ↔ this cluster's *residue in the human*).

## Status / next

Incubating. **Step zero:** recover Node B's encoder/artifact. Then per-node first cuts as above.
Individual runs bank their results (H0 vs observed, including nulls) under `design-docs/experiments/`.
