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
(tvi `output/thought-vault-integration-data/nv4096/`, 85,570/87,004 — tvi#43) and are
**untouched by analysis to date**.

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

**Named confound, unknown quantity:** natural language mediates between the source model
and the home re-embedding — the corpus embeds *text the models emitted*, re-encoded by a
different model. Raw material: `tvi/data/source/` (raw claude logs and the other five
channels).

## The two-ratchet prune filter (Measurement vs Orchestration)

**Definition not yet fully reconstructed** — only the name and one operating fragment
survive: *prune to the parts that ratchet the bootstrap*; work earns its keep by advancing
either the **Measurement** capability or the **Orchestration** capability, and a clean
NEGATIVE is a gain held (a path closed), not a failure. The filter's action is visible in
the record: Spike A logged FLAT/inconclusive, Spike B logged as *trustworthy NEGATIVE*
(HANDOFF §8). Operator words needed to complete this section.

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

Totally speculative until validated — the embeddings have not been touched by analysis.
Several projects are loosely roadmapped for the corpus (HANDOFF §8: Dest 1 corpus-mapping /
Dest 2 directional behavioral axes; residual-stream jolt rig as a later separate bet;
prompt-archaeology POC). tvi has been an **intake pool** rather than an operator priority
since ~Jan 2026 — material is tossed in deliberately, to be sorted as agent capability
becomes sufficient. That is a design choice, not neglect; this file is part of the sorting.

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
