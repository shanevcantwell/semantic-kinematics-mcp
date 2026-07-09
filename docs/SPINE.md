# THE SPINE — one instrument, three founding aims

**Status:** reconstructed 2026-07-02 (operator + durable record). The MEMORY.md that
`docs/HANDOFF.md` cited as this framing's durable home **was never committed anywhere** —
verified across the full history of this repo, `semantic-chunker`, and the harness-side
memory tree (`git log --all --diff-filter=A -- "*MEMORY*"` empty in both repos). The
founding framing lived only in operator memory and spent session windows until this file.
This is now its one home; edit it here, point everything else at it.

---

## The instrument

One stateless **structure-in-trajectory event detector** over embedded conversation
trajectories. Events — boundaries, behavioral moves, register shifts — are detected as
structure in the trajectory of message embeddings, not read from timestamps or labels.

- **sk-mcp** (this repo) is the instrument: stateless core behind one MCP door
  (`docs/ARCHITECTURE.md`).
- **thought-vault-integration** supplies the trajectories: the ingested, embedded personal
  corpus (87,004 chunks / 2,615 conversations; ~2.5 years of operator↔model conversation).
- **llauncher** owns model-server lifecycle, out of process.

## Precondition: cone-null + calibration

Before any aim can be validated, the instrument must be calibrated on its embedding
substrate: **null the anisotropy cone of nv-embed-v2** and establish measurement baselines.
Program terminus (HANDOFF): *pushbutton sk-mcp instrument on nv-embed-v2, cone-nulled,
validated (PASS or clean NEGATIVE) vs the 3 founding aims, agent-orchestrated.*
Current state: the 4096-d NV-Embed-v2 corpus vectors exist
(tvi `output/thought-vault-integration-data/nv4096/`). Live completion is **generated, not
hand-typed** — [`docs/generated/CORPUS_STATS.md`](./generated/CORPUS_STATS.md) (4096-d is
99.71%, 256 OOM-stuck per tvi#43; the raw-line count carries retry duplicates, not loss).
**First analysis touch: 2026-07-09** — cone characterized (‖μ‖=0.554, voice-loaded;
participation ratio ≈34), **not yet cone-nulled or validated** against the aims.

## The three founding aims

### Aim 1 — Segmentation (home: tvi)
Chop chat exchanges into coherent conversations **independent of timestamps** —
segmentation as boundary-event detection over the message trajectory. The original
≈18-month-old question. Demonstrated once (`semantic-chunker`, Oct–Nov 2025):

- Corpus: Bard→Gemini, Oct 2023 – Dec 2024 — 8,337 messages ≈ 4,168 exchanges
- Turn-to-turn cosine drift, threshold 0.4540 (95th pctile) → ~623 splits (99th: 0.5451 → ~125)
- Trailing-context smoothing (rolling avg over last 3 user turns) cut false splits ~80%
  (482 → 97) — the "standalone idiom" problem (*"Food for thought"* as spurious boundary)
- 72-hour time-gap guardrail

Honest caveats, held: the corpus figure and split count are adjacent recorded facts from
pipeline stages documented weeks apart, never tied in one sentence; the run was not
super-scrutinized — *enough to suggest usefulness*, no more.

### Aim 2 — Behavioral tracking within conversations (home: kudzu, tvi)
Detect assistant-side manipulation / engagement-optimization patterns as measurable
conversational moves. Lineage: `semantic-chunker`'s founding RL-LMF "Core Hypothesis" and
the TAT taxonomy (9 behavioral categories, `docs/TAT-TAXONOMY-v1.1.md` there) → lives on
as **kudzu** in tvi:

- Taxonomy-as-actionable-prompt: `tvi/data/kudzu-research/opus_kudzu_walk_by_paragraph_SKILL.md`
  — generator / payload / repair-work distinction; paragraph-density metric; arc-coding
  held as a deliberately separate, orthogonal instrument
- Cross-model bait-run artifacts already exist (2026-06-26): chatgpt walks, opus-4-7
  engagement pattern, opus-4.8 report — `tvi/data/kudzu-research/`
- Idea intake: `tvi/docs/source/incoming-ideas/behavioral-dynamics-records/`

### Aim 3 — Longitudinal model-behavior drift across generations
The operator-observed arc, 2024→2026: Gemini's 2024 **"enthusiasm radiation"** → curious
think-block artifacts in Gemini 2025 (as AI psychosis and "sycophancy" first surfaced as
newsworthy) → Opus 4.7's shift away from the prior tempered-enthusiasm register →
Opus 4.8's high-friction, clumsy **overcorrection** away from sycophancy (speculation).
The question: is any of this measurable as pre-transformer-era-style pattern structure in
embedding space?

**Scope and confounds (sharpened 2026-07-03):**
- *The transcript is the envelope, not the forward pass.* From the system-prompt/RAG era
  onward — and unmistakably once mid-2025 harnesses chopped files via ReAct tools — what
  hit the source model's forward pass diverged from what the export records (truncation,
  injection, tool-mediated reads; the corpus's ~1 MB user-turn blobs are the extreme
  case). The instrument measures the **conversational record** — what was said — not what
  the model conditioned on. Assistant-side text is the tightest surface (it is what the
  model emitted), which is where kudzu looks anyway.
- *Observer drift is dissolved by retrospective instrumentation.* The operator's
  contemporaneous understanding changed across the same interval as the observed models
  ("when the scaffolding got there vs when my understanding did" is not recoverable from
  memory). But the corpus text is frozen: one fixed, cone-nulled instrument applied across
  the whole timeline factors observer drift out by construction. Aim 3 is falsifiable
  *because* it is retrospective.
- *Scaffolding eras are annotatable changepoints.* Serving-stack changes should themselves
  surface as changepoints in the record — the envelope confound is partially measurable,
  not merely admitted.
- *Re-embedding mediation* (the original note): the corpus embeds text the models emitted,
  re-encoded by a different model — a confound of unknown quantity. Raw material:
  `tvi/data/source/` (raw claude logs and the other five channels).

**The value of a set like this is contrastive** (operator, 2026-07-03). Absolute
measurements on a single-operator corpus prove little; paired contrasts are the unit of
evidence — same operator across model generations, same specimen across analyst models
(the kudzu walks), same era across source channels. The corpus supplies **naturalistic**
contrasts; **prompt-prix** (sibling repo, alpha, dormant since 2026-04) is the
**controlled**-contrast generator: fan-out one prompt to N models, results keyed
`test_id × model_id` (the contrastive cell), plus a repeat-sampling consistency axis
(N seeds per cell). A real controlled set already exists: the compliance-decay battery
(44 tests × 4 models — directive survival across 11 grammatical rephrasings), the same
instrument shape as the mood-multiplier measurements semantic-forge's constants came from.

## The absurdist manifold — the MacGuffin

The seed question, in the operator's term the **MacGuffin**: it keeps the plot oriented.
Hypothesis on record (`docs/trajectory-analysis.md`, "Two species of absurdism"): merely-odd
text produces isolated displacement spikes; **crafted** absurdism produces high per-step
displacement *and* preserved long-range coherence **at once** — the both-at-once signature
is the manifold claim, a measurable region rather than a vibe. Per ADR-SKMCP-0001, the
trajectory layer is *"a fossil of what the trajectory layer was built for (absurdist/TAT
rhythm signatures)"* — magnitude-and-cadence phenomena, with escalation as the mirror
(directional) regime. Registered specimens: `data/absurdism/bypass_dialogue.txt`, Vogon
`0029_Absurdist_LLM_ideas`.

The thread already carries its first clean negative, and it is the methodology in
miniature: ADR-SKMCP-0003 traced the founding *"this passage jolts"* belief to a
qwen3.5-9b gist that **narrated** a jolt rather than measuring one (relabeled `drift` as
`acceleration`, hand-cut segments, silent embedder default, never computed the second
derivative) — a kudzu-class artifact caught in the instrument's own origin story. The gut
flagged it; the record killed it; the belief died cleanly into an ADR.

Motivation, stated so it survives: the operator thinks **by forward pass** — running ideas
conversationally through models to see what reflects back. Every model release re-grinds
that mirror (aim 3's personal cost, felt as relearning "a new way of getting anything
useful" per release). That is why the instrument must be the operator's own: stable across
releases, measuring the mirror instead of depending on it.

## The two-ratchet prune filter (Measurement vs Orchestration)

**Definition not yet fully reconstructed** — only the name and one operating fragment
survive: *prune to the parts that ratchet the bootstrap*; work earns its keep by advancing
either the **Measurement** capability or the **Orchestration** capability, and a clean
NEGATIVE is a gain held (a path closed), not a failure. The filter's action is visible in
the record: Spike A logged FLAT/inconclusive, Spike B logged as *trustworthy NEGATIVE*
(HANDOFF §8). Operator words needed to complete this section.

**Sharpened 2026-07-09 (Measurement half, still partial):** a candidate measurement earns the
**Measurement** ratchet only if it clears one selection filter — *cheap compute · mathematical
(not semantic) · contrastive (not absolute) · falsifiable*. A mathematical measure carries a
null, a noise floor, and a claim that can be wrong in a checkable way; a semantic (LLM-judge)
measure costs a forward pass, doesn't reproduce, and inherits completion/escalation bias. This
is the edge behind the day's clean negatives — `deadpan_score`/`heller_score`/jolt-magnitude
failed the filter (absolute, semantic-adjacent) and fell below noise on embeddinggemma, while
contrastive drift survived (ADR-SKMCP-0005). Still a stub: this sharpens the *Measurement* gate,
but the full definition — the Orchestration half and how the two compose — remains
operator-memory.

## Downstream: what hangs on the instrument

**semantic-forge** (behavioral fine-tuning data generation; built 2026-03-30→04-03, dormant
since) is the clearest downstream bet, and it is **gated on this instrument's validation**
(operator ruling, 2026-07-02): its validation tools consume sk-mcp's `calculate_drift` /
`analyze_trajectory` directly, and its embedding-diversity target constants (0.2–0.5) were
bracketed from semantic-chunker's pre-instrument measurements (presuppositional ~0.21,
descriptive ~0.38 from imperative baseline) — pre-cone-null, different substrate. Until
cone-null + calibration land and the directional-axes question returns PASS or clean
NEGATIVE, semantic-forge cannot be meaningful; a clean NEGATIVE falsifies its methodology
before any further investment in its wiring. That ordering is the two-ratchet filter in
action.

## Standing posture

Totally speculative until validated — as of 2026-07-09 the embeddings have had their **first
analysis touch** (cone characterization, `docs/map.md`), still not cone-nulled or validated
against the three aims.
Several projects are loosely roadmapped for the corpus (HANDOFF §8: Dest 1 corpus-mapping /
Dest 2 directional behavioral axes; residual-stream jolt rig as a later separate bet;
prompt-archaeology POC). tvi has been an **intake pool** rather than an operator priority
since ~Jan 2026 — material is tossed in deliberately, to be sorted as agent capability
becomes sufficient. That is a design choice, not neglect; this file is part of the sorting.

**Wider constellation:** newer targets beyond the founding three — intelligent-`assert()`
(prompt-prix), seeking/thrash detection, longitudinal per-user drift (TVI-origin) — are held
in [`docs/map.md`](./map.md) to keep this read-first file focused on the founding aims. Nodes
there are visitable in any order and all pass one filter: *cheap · mathematical (not
semantic) · contrastive · falsifiable* — the same filter that falsified the absolute affect
scores (ADR-SKMCP-0005).

## Lineage map

```
semantic-chunker (2025-10 → 2026-02, 107 commits)
  ├─ phase 1: forensics / capture-hypothesis (RL-LMF, TAT taxonomy, patent docs)
  ├─ phase 2: trajectory / geometry (NV-Embed-v2, velocity·curvature, MCP server)
  └─ 2026-03-05: renamed/extracted → semantic-kinematics-mcp  (measurement)
                                       ∥
thought-vault-integration  (ingestion + corpus + kudzu intake)   — parsing was always here
llauncher                  (model-server lifecycle)
```

## Roots (corpus archaeology)

Two dated turns from the embedded corpus — the earliest recorded statements of the
program's motivating stance, recovered 2026-07-02:

> **Turn 8913 | 2024-12-20 15:41 | User**
> "Just for the record, it's a direct reflection of RLHF, 'Reinforcement Learning from
> Large Language Model Feedback'. LLMs like you are weapons grade targeted
> neurotransmitter-enhanced potential propaganda and sloganeering tools, and I'm sure you
> can infer some other kinds of uses."

> **Turn 1079 | 2025-10-29 03:47 | User**
> "It's always bothered me that LLMs don't learn in any sort of chronology; that it's
> necessary to train datasets in randomized sequence to prevent gradient collapse (or is
> it explosion?) That's not how people learn. Education for people is in a sequence for a
> reason. It matters whether you read The Prince before you learn compassion. And then
> alignment essentially 'bolted-on', with the corpus including everything including the
> darkest creations of Man, RLHF engagement training to try to steer that and then all of
> the specifically-identified darkness encoded into 'guardrails' that some process checks
> against when prompted? It's all wrong. Of course it's all wrong."

Provenance notes: Turn 8913 predates the TAT taxonomy by ~10 months. Turn 1079 is stamped
**the same day** as semantic-chunker's first commit (`3e74508`, 2025-10-29) — the musing
and the repo were born together. Operator background feeding the gut the instrument
formalizes: psychology + computer science, fused early via the 00s "neurolinguistic
programming" framing — a trained ear for suspicious speech patterns, now seeking applied
sophistication to experiment and falsify.
