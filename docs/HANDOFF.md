# Cross-Repo Design Handoff Index

**Written:** 2026-06-08  
**Updated:** 2026-06-11 — ADR-002 and ADR-003 promoted to Accepted; BulkEmbedder
merged to main (PR #12); ADR-002 migration tracked in sk-mcp #11.  
**Branch at time of writing:** `docs/handoff-index` (off `feat/axis-alignment`)

---

## 1. Purpose / how to use this doc

This is the single entry point for resuming a multi-repo design effort that
produced three ADRs, one working code branch, and five GitHub issues across
three repositories. Read this document first to orient yourself, then follow
links into the ADRs and issues for detail. The ADRs and issues are the
authoritative sources; this index only summarizes and connects them.

---

## 2. The big picture

The effort is building a semantically honest, reproducible embedding analysis
pipeline that spans three repos. **`semantic-kinematics-mcp`** (sk-mcp) is the
public toolkit; it provides the embedding adapters, analysis tools (MCP server),
and now a bulk-embedding engine. **`thought-vault-integration`** is the personal
corpus manager; it produces the real-conversation text that becomes both training
signal and the empirical null distribution for axis-alignment significance tests.
**`llauncher`** manages the local model-server process lifecycle.

The three ADRs interlock in a specific order of dependency: ADR-002 (unified
adapter) must land before ADR-001's null cache can receive vault-produced
embeddings, and ADR-003 (stateless MCP) depends on ADR-002's per-call adapter
resolution. **ADR-002 and ADR-003 are now Accepted** (PR #10 promoted ADR-003 on
2026-06-08; ADR-002 followed on 2026-06-11, its `model_name` format settled
jointly with ADR-003 Resolution 1). ADR-001 remains Proposed pending real-corpus
validation. ADR files stay under `ADRs/proposed/` to preserve cross-repo links;
the Status field in each file is authoritative.

---

## 3. Design decisions (ADRs)

| ADR | Decision (one line) | Branch | Origin SHA | Status |
|-----|---------------------|--------|------------|--------|
| [ADR-001](ADRs/proposed/ADR-001-referential-axis-alignment.md) | Add `analyze_axis_alignment` tool: project sentences onto an anchor-defined axis, z-scored against a corpus null | merged to main | — | Proposed |
| [ADR-002](ADRs/proposed/ADR-002-unified-embedding-adapter.md) | One `EmbeddingAdapter` ABC shared by sk-mcp and thought-vault; `BulkEmbedder` wraps any adapter for corpus-scale work | merged to main | — | **Accepted (2026-06-11)** |
| [ADR-003](ADRs/proposed/ADR-003-stateless-mcp-contract.md) | sk-mcp MCP tools become stateless (model selection per-call); `model_load`/`model_unload` removed; lifecycle moves to llauncher | merged to main (PR #10) | — | **Accepted (2026-06-08)** |

### ADR-001 — Referential axis-alignment analysis

Adds an `analyze_axis_alignment` MCP tool that measures whether a passage
"marches" along a user-defined semantic axis (e.g. escalation). The axis is
defined by positive/negative anchor exemplars; significance is a z-score against
an empirical null distribution from a background corpus projected onto the same
axis. The shuffle null is explicitly excluded (net displacement is
interior-order-invariant). Implementation is already on `feat/axis-alignment`
(`mcp/commands/axis_alignment.py` + `scripts/build_axis_null.py`).

Open questions left by ADR-001: which of the three readouts (position trace,
axis drift, axis-restricted straightness) leads the headline result; null-cache
manifest schema as it hardens (staleness detection beyond model-name match);
ADR numbering scheme (`ADR-001` vs. `ADR-CORE-NNN`).

### ADR-002 — Unified embedding adapter / BulkEmbedder

Consolidates two independently-drifted embedding implementations (sk-mcp's
`EmbeddingAdapter` ABC and thought-vault's `EmbeddingBridge`) into one shared
abstraction living in sk-mcp. The vault will depend on sk-mcp via
`pip install -e`. The `BulkEmbedder` wrapper (merged via PR #12) adds
checkpoint/resume, sub-chunking, and token-aware batching to any adapter. It is
**deliberately retry-free**: in-engine backoff hides server failure patterns;
recovery is idempotent re-invocation over checkpoint resume, owned by the
supervising layer (agent/operator). The vault's `EmbeddingBridge`
is deleted at cutover (no shim — see #11 ruling). Critically,
`model_name` becomes the underlying model identity (not a backend-prefixed
label), which allows vault-produced embeddings to key into ADR-001's null cache.

~~Open questions left by ADR-002~~ — resolved at acceptance (2026-06-11, see the
ADR's "Resolved decisions" section): canonical `model_name` is llauncher's
string (e.g. `embeddinggemma-300M-F32`), transport metadata is adapter
construction detail; `BulkEmbedder` lives beside the adapters
(`embeddings/bulk.py`, merged via PR #12); migration ordering affirmed and
tracked in sk-mcp #11. Still deferred: dtype/precision unification; ADR
numbering across the two repos.

### ADR-003 — Stateless MCP control-plane

Removes server-side model state from sk-mcp. Each MCP tool call receives model
selection (`backend`/`model`/`base_url`, env fallback) and resolves a fresh
adapter; no adapter is retained between calls; the cross-call embedding cache
is dropped. `model_load` and `model_unload` are removed; process lifecycle moves
entirely to llauncher. The Gradio UI must own its own client-side cache for
reactive slider behavior. The nv_embed path (SentenceTransformers/PyTorch, not
GGUF) requires llauncher to add a vLLM/SentenceTransformers server type before
sk-mcp can route it through llauncher — tracked in llauncher #155.

~~Open questions left by ADR-003~~ — resolved at acceptance (2026-06-08, PR #10,
see the ADR's "Resolved decisions" section): per-call shape is `model_name` +
`base_url` with env fallback (no backend enum in the call identity);
`StateManager` becomes a thin stateless resolver (fields/mutator/cache removed,
class kept). Still open: sequencing of llauncher vLLM/SentenceTransformers
expansion (llauncher #155) relative to sk-mcp's cutover — partial cutover
(llama-server stateless first, nv_embed follows) is the working assumption.

---

## 4. Code — BulkEmbedder engine (merged)

**Merged to main 2026-06-11 via PR #12** (was `feat/embedding-engine`). A
pre-merge code review surfaced and fixed: k==1 vectors now L2-normalized like
multi-chunk ones (all stored embeddings unit-norm — load-bearing for ADR-001
z-scores); checkpoint file handle opened inside try/finally and only when work
is pending; corrupt checkpoint lines skipped per-line instead of discarding all
completed work; over-budget single items warn. 13 BulkEmbedder tests passing.

Three files:

- `semantic_kinematics/embeddings/bulk.py` — `BulkEmbedder` class (303 lines):
  wraps any `EmbeddingAdapter`; texts within token limit pass through whole;
  oversized texts are sentence-split and sub-chunk vectors are averaged back to
  one L2-normalized vector; cross-text packing fills batches up to
  `max_tokens_per_request`; checkpoint/resume mirrors thought-vault `_failed`
  semantics (a vector must be right-dimensioned, non-zero, and not `_failed` to
  count as done; bad entries are retried on resume; group-embed failures are
  isolated rather than aborting the corpus).
- `scripts/embed_corpus.py` — CLI entry point (86 lines).
- `tests/test_bulk_embedder.py` — offline `FakeAdapter`-based tests (193 lines).

Status: merged, review-hardened, **not yet run against the real corpus**. The
real validation run (sk-mcp #3) is now unblocked and is the next concrete step.

---

## 5. Tracked open work (issues)

| Issue | Title | What it unblocks |
|-------|-------|------------------|
| [sk-mcp #2](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/2) | Implement ADR-003: stateless MCP control-plane | sk-mcp MCP surface becomes reproducible and self-contained; prerequisite for clean multi-session use; now scoped against the Accepted ADR-003 resolutions |
| [sk-mcp #3](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/3) | Run chat-log corpus through BulkEmbedder; validate ~45 min throughput | Validates BulkEmbedder on real data; produces embeddings that serve as ADR-001 axis-alignment null. **Unblocked by PR #12** |
| [sk-mcp #9](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/9) | UI bypasses the MCP contract (direct `mcp.commands.*` imports) | UI and external callers go through one door; sequenced after #2 (needs the per-call param shape in place) |
| [sk-mcp #11](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/11) | Implement ADR-002: unified embedding adapter migration | Adapter generalization, `model_name` canonicalization (null-cache-invalidating), normalization audit, vault cutover — the gate between #3's embeddings and the ADR-001 null cache |
| [thought-vault #28](https://github.com/shanevcantwell/thought-vault-integration/issues/28) | Re-extract corpus from richer sources and re-embed | Richer signal (full HTML/markdown, per-message granularity); clean embedding base for all downstream analysis |
| [thought-vault #29](https://github.com/shanevcantwell/thought-vault-integration/issues/29) | Reproducibility: capture embedding-server config; consolidate bulk runner | Documents llauncher `extra_args` (`--embeddings --log-disable`, ubatch/batch 4096) in-repo; decides fate of uncommitted supervisor script |
| [llauncher #155](https://github.com/shanevcantwell/llauncher/issues/155) | Add vLLM/SentenceTransformers server type for non-GGUF embedding models | Lets llauncher own nv_embed (4096-d) lifecycle; prerequisite for ADR-003's full stateless nv_embed path |

---

## 6. Open questions / unresolved

- **ADR numbering scheme.** The ADRs use project-local `ADR-001/002/003` but an
  `adr-namer-draft.sh` script suggests a cross-repo `ADR-CORE-NNN` scheme.
  Which wins, and does it affect the thought-vault's own `ADR-003/004` numbering?

- ~~**Canonical `model_name` format.**~~ Resolved 2026-06-11 (ADR-002
  Resolution 1, jointly with ADR-003 Resolution 1): llauncher's canonical
  string, e.g. `embeddinggemma-300M-F32`; transport metadata is adapter
  construction detail. Migration of legacy-keyed caches tracked in sk-mcp #11.

- ~~**Per-call parameter shape for stateless tools (ADR-003).**~~ Resolved
  2026-06-08 (ADR-003 Resolution 1): `model_name` + `base_url`, env fallback
  (`EMBEDDING_MODEL` / `EMBEDDING_SERVER_URL`); no backend enum in call identity.

- ~~**`StateManager` fate (ADR-003).**~~ Resolved 2026-06-08 (ADR-003
  Resolution 2): thin stateless resolver — class kept, retained fields, mutator,
  and cross-call cache removed.

- **Which readout leads `analyze_axis_alignment` (ADR-001).** Three candidates:
  position trace z-scores, axis drift (net signed march), or axis-restricted
  straightness ratio. Not decided; depends on real-corpus validation.

- **Null-cache staleness beyond model-name match (ADR-001).** Current guard
  refuses a null whose `model_name` differs from the active adapter. Whether to
  add a date/hash-based staleness signal is deferred.

- **Normalization audit (ADR-002).** The LM Studio adapter currently does not
  L2-normalize; embeddinggemma via llama-server behavior is not yet verified. ADR-002
  requires each adapter to declare its normalization contract explicitly; the
  audit has not been done. Now tracked as sk-mcp #11 item 3. (Mitigation already
  in place: `BulkEmbedder` L2-normalizes every stored vector as of PR #12.)

- **Sequencing: llauncher #155 vs. sk-mcp ADR-003 cutover.** sk-mcp can go
  stateless against llama-server backends immediately; the nv_embed (PyTorch)
  path through llauncher requires #155 first. Is the partial cutover (llama-server
  stateless now, nv_embed follows) acceptable, or should both wait?

- **thought-vault #29: supervisor script + packing-ceiling edit.** An uncommitted
  `scripts/run_embeddings_supervised.sh` and a bridge throughput edit (packing
  ceiling 1500) on branch `fix/embedding-checkpoint-resume` are likely superseded
  by BulkEmbedder but have not been formally closed or discarded. PR #27
  (checkpoint-resume fix) is still open.

- ~~**BulkEmbedder module location (ADR-002).**~~ Resolved 2026-06-11 (ADR-002
  Resolution 2): stays beside the adapters at
  `semantic_kinematics/embeddings/bulk.py`; merged via PR #12.

---

## 7. Suggested next actions (ordered, updated 2026-06-11)

Done since the original index: ADR-003 accepted (PR #10), ADR-002 accepted,
BulkEmbedder merged with review fixes (PR #12), ADR-002 migration issue filed
(sk-mcp #11). The `model_name` format is settled — corpus embeddings written
from here forward should use the canonical identity string.

**Dataset path (critical chain for thought-vault):**

1. **Capture and commit embedding-server config** (thought-vault #29). Before the
   run, document the required llauncher `extra_args` (`--embeddings
   --log-disable`, batch/ubatch 4096) so the run is reproducible. Decide the
   fate of the supervisor script and the vault's #27 PR (likely superseded by
   BulkEmbedder).

2. **Run the corpus through BulkEmbedder** (sk-mcp #3). Validates the engine on
   real data (~80K messages / ~39K parsed chunks — granularity to reconcile;
   ~45 min target) and surfaces tuning needs before ADR-002 migration work.
   Gated by Spike 1 (#20 fix) — see §8; until #11 item 2 lands, these embeddings
   are keyed by the current adapter's `model_name` and are validation output,
   not yet null-cache-eligible. Engine core is smoke-validated live (2026-06-11,
   :8082): 21/22, unit-norm, semantic sanity good, resume idempotent.

3. **ADR-002 migration** (sk-mcp #11): generalize the OpenAI-compatible adapter;
   canonicalize `model_name` (null-cache-invalidating — rebuild/remap legacy
   caches); audit normalization per backend; cut the vault over via the
   `EmbeddingBridge` shim.

4. **Re-extract and re-embed corpus** (thought-vault #28) through the shared
   adapter at per-message granularity, with canonical `model_name`.

5. **Build the ADR-001 axis-alignment null** from the re-embedded corpus
   (`scripts/build_axis_null.py`); validate `analyze_axis_alignment` against
   real signal. This is also the natural point to resolve ADR-001's remaining
   open questions (leading readout, null binding — sk-mcp #1) and promote it.

**Architecture track (parallel, does not block the dataset path):**

6. **Implement ADR-003** (sk-mcp #2) per the accepted resolutions: per-call
   `model_name` + `base_url`, thin-resolver `StateManager`, delete
   `commands/model.py`, UI caching client-side, fix `embed_text`'s decorative
   `model` argument. llama-server backends go stateless now; nv_embed follows
   llauncher #155.

7. **UI conformance** (sk-mcp #9), sequenced after #2 — route the Gradio UI
   through the MCP contract instead of direct `mcp.commands.*` imports.

8. **llauncher #155** (vLLM/SentenceTransformers server type) — prerequisite for
   the fully stateless nv_embed path; parallelizable with 6–7.

---

## 8. Target state & spike plan (rev 2026-06-15)

### §8 rev 2026-06-15 (c) — residual-stream pivot; contrastive instrument; magnitude rehabilitated
- Spike B axis-free magnitude (sentence atom) = trustworthy NEGATIVE (PR #32). North star recorded: **residual stream of a LOCAL model** (NOT Claude internals/CC logs — inaccessible / thinking-not-retained).
- Deep-research survey: `docs/research/residual-stream-jolt-survey.md` (arXiv-grounded). Infra finding: web MCP tools (WebSearch/WebFetch, web_search/web_fetch, consult_advisor) were permission-denied/unregistered to the subagent → it grounded via curl to arXiv. Gap to fix; also blocks identifying the JetBrains post-training-checkpoint model (news lookup, not arXiv) — PENDING.
- KEY correction to our own framing: **magnitude is not dead, it was mis-atomed.** StALT (Furuya & Tanimura 2026, arXiv:2605.01853) — layer-weighted inter-TOKEN residual transition magnitude — separates reasoning quality. Converges with the token-wise hypothesis; means we do NOT need a projection axis (Spike A flat no longer blocks the bearing instrument). Carry next-token entropy ("forking tokens", Wang 2025, arXiv:2506.01939) + linear correctness probe (0.95 AUROC vs 0.59 surface, Yuan 2026, arXiv:2605.09502) alongside; across-LAYER prob trajectory is flat (Kim 2025, arXiv:2507.06722) → observable-specific, not free.
- Contrastive instrument design (user): base vs instruct = differential isolating post-training; JetBrains post-training checkpoint staircase = derivative (jolt-structure emergence). Asset state: instruct HF on host (`/mnt/storage/LLMs/google/gemma-4-31B-it-assistant`, `gemma-4-26B-A4B-it-assistant`); 31B base on host is GGUF-only; base HF on the Windows node.
- Access ladder (serving != activation access): (1) next-token entropy via served endpoint — node-serving enables NOW, verify prompt-token logprobs/echo; (2) final-layer token vectors via pooling=none; (3) full per-layer StALT via HF + NNsight hooks = the rig (31B bf16 ~62GB > 48GB 8000 → de-risk on small R1-distill [has CoT-correctness labels] or 26b-a4b MoE first; co-locate base+instruct HF).

### §8 rev 2026-06-15 (b) — ADR-SKMCP-0002 tool contract; Spike A run; IMPLEMENTATION START POINT
**Canon now: ADR-SKMCP-0002** (`docs/ADRs/proposed/ADR-SKMCP-0002-bearing-analysis-tool-contract.md`, PROPOSED) on branch `docs/adr-skmcp-0002` (`c7f86d1`, **PR #27**). Operationalizes ADR-SKMCP-0001 (0001 = the math; 0002 = how it ships); extends the ADR-001 build-validated-artifact + one-call-consume pattern (`build_axis_null.py` → `analyze_axis_alignment`) into the bearing regime.
**The contract** — `bearing_analysis` MCP tool **family** (lifecycle-named so a non-frontier agent reads the order off the tool list):
- `initialize_bearing_analysis` (patient): embed anchor grid → **validate axis** (= Spike A: union-SVD + coherence + length-confound → verdict `one-axis|n-axes|UNRESOLVED|flat`) → build **measured-displacement null** → emit self-describing artifact. Returns artifact ref + verdict.
- `run_bearing_analysis` (fast): project displacement vs artifact → signed component (σ) + residual (σ) + cosine. **Embedder is PINNED by the artifact** (agent passes only `{displacement, artifact_ref}`) → enforces 0001 single-embedder rule + honors #14 (no silent default) + minimal call surface.
- `get_bearing_analysis_status` (inspect; the one droppable-for-v1 piece).
- **Self-describing artifact = load-bearing object** (embedder identity, axis-validation verdict, null protocol + self-validation diagnostics, provenance). **Five cross-cutting rules as REFUSALS**: null-exemption typing · embedder-*validated* not just labeled · circularity guard · falsification-shaped confidence · target-state forms (post ADR-003/#2 stateless per-call `model_name`+`base_url`, ADR-002/#11 canonical `model_name`).
**Spike A RAN** (`scripts/spike_a_escalation_svd.py`, :8082 embeddinggemma, `escalation_grid.yaml` 24 pairs + 4 mood): **FLAT** — union-SVD top comp 14% raw / 12% centered, no cliff; within-axis coherence 0.10–0.17 (deltas barely cohere); inter-axis means near-orthogonal but that's hi-d noise *given* the low within-axis coherence; length-confound r=0.37 char / 0.19 token (moderate); mood only contempt→tone (0.24). **Not a coherent escalation axis as instrumented.** Attribution (embedding-null vs grid-too-weak-at-N8 vs embedder-captive) UNRESOLVED → becomes `initialize`'s job, not a separate experiment. Raw: `/tmp/sk_spike_a/results.json` (ephemeral); script now committed to main (PR #28).
**North star (user, FIRM):** target = dead-simple **1-step reproducible** tooling producing confident/accurate results whose *action* reproduces — NOT the data while developing (good data kept, never discarded). "No matter how many steps to be confident, not done until reproducible as 1 step." The agent is NOT the integration layer. Primary caller = **non-frontier agent** → fast-interaction hot path + endless-patience setup; names teach the workflow; errors instruct.
**Grounding done (3 layers):** README (intent) + code-explore (reality) + incoming-ideas deflation (sibling `design-docs` repo; 5 rules extracted, zero contradictions w/ 0001; all Opus-drafted self-deflating → trust the operational tests not the prose). **README DRIFT LIST captured (separate deliverable, NOT yet actioned):** 9→7 tools post-ADR-003; `embed_text` `model` param decorative (ignored, uses StateManager); `BulkEmbedder` + `embed_corpus.py` undocumented; `model_name` backend-prefixed (pre-#11 canonicalization).
**>>> IMPLEMENTATION START POINT (begin here next session) <<<**
1. **`initialize_bearing_analysis` FIRST** — absorbs the Spike A logic that already runs (promote it from the one-off script into core behind the tool) + builds the measured null from **consecutive within-conversation deltas** of the vault (`thought-vault-integration/output/vectors/`, 80,520 turns embeddinggemma 768d) with self-validation (convergence vs subsample N, bootstrap CIs, distribution shape). NOT random cross-conversation pairs. Emit the self-describing artifact.
2. **`run_bearing_analysis`** — fast; embedder-from-artifact; coefficient-correction-shaped machine-legible output; instructive errors.
3. **`get_bearing_analysis_status`** (or drop for v1).
4. Build against **target forms** (ADR-003 stateless, ADR-002 canonical `model_name`) — do NOT build on `model_load`/`model_unload` or backend-prefixed names. Test coverage per the project rule.
   Open Qs (ADR-0002): null summary shape (empirical-quantile if non-Gaussian) · `cosine` exemption · status-tool-in-v1 · position-regime rename (`initialize_axis_analysis`/`run_axis_analysis`) · forge schema.
**Also pending (user-requested, after this):** refine the **git subagent definition** via a cross-repo explore of the last week's Claude Code chat logs (apply accumulated knowledge). Data point for it: the git subagent misreported a base SHA this session (said `c671518`, actual merge tip `cf82241`; branching was correct, prose label wrong).
**Issue #25:** Spike A item = DONE-as-pipeline (flat fixture); items (1) Vogon run, (2) B-cluster mapping, (4) primitive build now subsumed under ADR-0002 implementation; (5) #14 defaults still deferred; (6) PR #26 merged + branches/worktree cleaned ✓.
**Git state (session close):** sk-mcp `main` @ `07803d1` (PRs #26 seam/nomic · #27 ADR-0002+handoff · #28 Spike A script · #29 gitignore `.claude/`); harness-tools `main` @ `f777d96` (#32 git-agent def refinements · #33 host-config canonical). Spike A script tracked at `scripts/spike_a_escalation_svd.py`.
**Central-repo + symlink discipline extended to Claude host config (sk-mcp = PILOT):** `~/.claude/system-prompt.md` and `sk-mcp/.claude/CLAUDE.md` are now gitignored symlinks into `harness-tools/claude/host-config/` (canonical); 3 stale orphan `.claude/systemPrompt.txt` removed (sk-mcp, semantic-forge, langgraph). Pattern to extend to other repos' per-project CLAUDE.md when desired. Git-agent definition refined from mined session evidence (ref-labeled verified SHAs, live-state derivation, gh/hook allowlist).
**README accuracy pass:** corrected aspirational "stateless core" (it's stateful pre-ADR-003) + inert `embed_text` model param; documented BulkEmbedder + `embed_corpus.py`.

### §8 rev 2026-06-15 — Stage-1 jolt smoke + seam; nomic fossil; vault confirmed
**Branch `fix/embeddings-nomic-default`** (off main, PR #26): `a2ee10e` trajectory seam + smoke harness; `1f1bba5` nomic silent-default hard-fail (#14 partial).
**Bearing primitive (ADR-SKMCP-0001) confirmed NOT implemented** — only ADR-001 axis-alignment exists. Seam now in place: `TrajectoryAnalyzer.analyze_embeddings(matrix)` / `analyze_segments(list)` (no spaCy split) — Spike B plugs in here.
**Stage-1 jolt smoke** (`scripts/smoke_jolt.py`, motivating fixture, NOT the real instrument):
- Specimen A — Adams bypass "Dent/Prosser" dialogue, sentence atom, embeddinggemma :8082 → FLAT (max_accel 0.217 @ idx43 = narration "…shadow over Arthur Dent's house"; 0 isolated spikes; deadpan 0.40). Punchlines ("With a torch.", "Beware of the Leopard") did NOT spike.
- Specimen B — escalation conv `e7c2fe94…`, precomputed per-turn vault vectors → strong isolated spikes (max_accel 0.598 @ idx147 = "O(2^n) complexity" topic-shift turn; 13 spikes; isolation>0.9 cluster idx41-64; deadpan 0.75).
- Reading: magnitude ALONE separated A/B (opposite of shared-signature premise) BUT confounded (A is bypass dialogue not Vogon poetry; sentence vs turn atom). Peaks on UNpredicted sentences = argument FOR bearing.
**Provenance:** original "this passage jolts" belief = qwen3.5-9b gist (https://gist.github.com/shanevcantwell/6c0344db773e11fce23591967f2e4572) that NARRATED a jolt — relabeled `drift` as acceleration, hand-cut segments, nomic via silent default (#14). Never ran 2nd-derivative. Not a measurement.
**thought-vault-integration CONFIRMED fully embedded:** `output/vectors/` — chunks.jsonl (80,520 per-turn) + embed_checkpoint.jsonl (80,520/80,520, embeddinggemma 768d, unit-norm). Dest1 embed prereq (vault #28) appears satisfied. Specimens: escalation `e7c2fe94…`, Vogon `0029_Absurdist_LLM_ideas`.
**Env provisioned:** pynvml→nvidia-ml-py (test noise gone); spacy 3.8.14 + en_core_web_sm 3.8.0. Tests 41→48.
**NEXT (fresh stab — issue #25):** (1) real Vogon-poetry run sentence-wise; (2) map B spike cluster to turn text; (3) Spike A union-SVD on escalation_grid (READY w/ caveats, N=8 floor); (4) build ADR-SKMCP-0001 primitive on analyze_embeddings seam vs measured-displacement null, 2 embedders; (5) deferred #14 defaults; (6) merge/clean PR #26; cleanup stray .claude/worktrees/agent-aa3af4a0bc84ae574.

---

## 8. Target state & spike plan (rev 2026-06-13)

> Supersedes the first 2026-06-13 cut, which scoped the terminal state too
> narrowly — it treated ADR-001's `analyze_axis_alignment` tool as the
> destination. **ADR-SKMCP-0001** (directional projection primitive, now promoted
> into this repo at `docs/ADRs/proposed/`) shows that tool is a *waypoint*, and
> that two destinations were chained as one. They share embedding infrastructure
> but fork. The standing decision (2026-06-13) is **falsify-first**: the bearing
> destination's viability is a cheap go/no-go that gates whether the expensive
> infrastructure is worth building at all.

### The noise, named: two regimes (different atom, null, tool)

The words "projection", "axis", and "null" each named one thing in each of two
regimes, which is what made the system feel noisy.

| Regime | Question | Atom | Null | Tool |
|---|---|---|---|---|
| **Position / rhythm** | "where does this sit / how does it drift?" | sentence (point) | **corpus-null** — where real text *sits* on the axis | ADR-001 axis-alignment; velocity/curvature (inherited) |
| **Bearing / motion** | "which way did this *move* vs a named axis?" | **displacement** (+ anchor pair) | **measured-displacement-null** — what random *motion* looks like, given anisotropy | ADR-SKMCP-0001 projection primitive |

The null collision is the sharp one: position-null is about where points *sit*;
displacement-null is about whether a motion's bearing *beats chance* in a
cone-shaped (anisotropic) space. Not interchangeable; each binds to its atom.
Diagnosis from ADR-SKMCP-0001: the reflexive angle measures (drift cosine,
curvature) **saturate near π/2 in high-d and discriminate nothing** — but the
same dimensionality *sharpens* a referential projection against a fixed axis.
That asymmetry is why the bearing regime is the real instrument and the
velocity/curvature layer is the rhythm-regime fossil.

### Two destinations (shared infra base, forked goals)

- **Dest 1 — map the thought-vault corpus** (position regime): drift/position
  over the personal corpus; ADR-001 corpus-null; **needs the full vault embedded
  at scale** → infra-heavy (#11, #16, #17, #20, resolver).
- **Dest 2 — measure behavioral axes directionally** (bearing regime; escalation
  → semantic-forge judge calibration): ADR-SKMCP-0001 projection; the axis comes
  from a curated **anchor grid, not the vault**; calibration data is model
  completions, **not the vault**. Needs almost none of the at-scale infra.

**Dest 2 leads.** Spikes A→B are go/no-go gates on the whole bearing enterprise
and cost almost nothing (anchor texts + current embedder + numpy). Industrialize
the vault embed (Dest 1) only with confidence the signal exists — fail loud and
fast first.

### Dependency shape

```
SHARED INFRA BASE:  resolver(#2/#14/#15) · #16 · #17 · #20(Spike C) · #11
  ├─ Dest 1 (vault mapping):    base → corpus embed at scale → ADR-001 corpus-null
  │                             → drift/position analysis
  └─ Dest 2 (behavioral axis):  [Spike A] escalation-axis SVD → [Spike B] projection
                                + measured-displacement-null → ADR-SKMCP-0001 implemented
                                → semantic-forge judge calibration
                                (needs almost none of the base; A/B run NOW on current tooling)
```

Off both critical lines (parallelizable): **#9** (UI/MCP contract, needs #2's
param shape); **llauncher #155** (gates only the nv_embed path).

### Spike A — Escalation-axis dimensionality (union SVD) — *bearing; falsify-first; prereq for B*

**Question:** is escalation one coherent direction or three separable axes
(tone/urgency/importance)? Until this resolves, "project onto the escalation
axis" has no defined target.
**Precondition (open input):** an escalation **anchor grid** — level↔escalated
phrasing pairs across the three sub-axes + mood variants. *Not locatable in the
ecosystem as of 2026-06-13* (grep hit mood/cathedral docs, no anchor-set
artifact). Spike A's first checkpoint is **locate-or-author the grid.**
**Method:** embed the grid in ONE embedder (embeddinggemma :8082 first); form
delta vectors (escalated − level); SVD; read the singular-value spectrum. Repeat
on a second embedder to test stability.
**Fail loud / kill criteria:**
- Flat spectrum / no dominant direction → escalation isn't a linear axis here → **STOP**.
- "Two strong + one marginal" → per the ADR's own honesty rule this is
  **UNRESOLVED, not a finding** → do not build on an ambiguous axis.
- Coarse structure disagrees across embedders → result is embedder-captive →
  that *is* the finding, and it gates the whole bearing enterprise.
**Output:** number of coherent axes + coarse-vs-marginal honesty call +
embedder-stability verdict. Go/no-go for Spike B. *Needs none of the infra base.*

### Spike B — Projection primitive + measured-displacement-null — *bearing; depends on A*

**Question:** does projecting a displacement onto the validated axis produce
signal that clears a measured null — i.e. does the instrument detect escalation
at all?
**Method:** implement the primitive (displacement → signed component, orthogonal
residual, cosine alignment); build the null by **resampling real-text
displacement deltas** projected onto the axis (mean/std) — *measured, not 1/√d*,
to inherit the space's anisotropy; project known escalated-vs-level test
displacements; compute sigma above null.
**Fail loud / kill criteria:**
- Known-escalated displacements don't separate from the null at meaningful sigma
  → the instrument doesn't detect its target → **STOP**.
- Large residual on known-escalated (motion mostly orthogonal to the axis) → the
  axis isn't capturing escalation → re-derive the axis (back to A).
- Null so wide nothing clears it → anisotropy defeats the measurement in this
  embedder → loud fail; reconsider embedder (A's embedder-choice finding).
**Output:** signed-component-vs-null sigma for known cases; validated null
protocol; go/no-go on the primitive → whether ADR-SKMCP-0001 graduates from
proposed toward implemented. *Still needs none of the at-scale infra.*

### Spike C — Tokenization calibration — *infra; = former Spike 1; #20*

Unchanged (see §7 / #20). Gates Dest 1's at-scale embed; **independent of A/B**.
Mechanism proven via the `/tokenize` probe; residual unknowns are corpus-level
distribution + pre-count throughput at ~39K scale.

### Spike D — Eligibility, regime-typed — *infra; = former Spike 2, reframed*

The hand-faked end-to-end proof against `build_axis_null.py`, but now it must
declare **which null** (corpus vs measured-displacement) and **which atom**
(sentence vs displacement) it validates — and the #16 checkpoint header must
record **regime + atom + embedder**, not just `model_name`. The reframing forces
the two-null distinction into the artifact so the regimes stop tangling.

### What is implementation, not spike

R (resolver seam, #2/#14/#15), #17 (normalize-before-average + non-uniform-scale
regression test), and #11 are execution — shape known, waiting on a spike finding
(R's canonical-string format ← Spike D) or direct work.

### Numbering note

ADR-SKMCP-0001 keeps its `SKMCP-NNNN` identity in this repo (cross-repo refs
depend on it) rather than renumbering to local `ADR-004`; the `ADR-00N` vs
`ADR-SKMCP-NNNN` reconciliation is the one open ADR-numbering thread.

### Sequence

**[Spike A] → [Spike B]** (falsify-fast, now, current tooling) → go/no-go on the
bearing destination. The infra track (Spike C, D, then #17/#16/#11.2/R) proceeds
for Dest 1 **in parallel only to the extent vault-mapping has independent
value** — otherwise it waits on A/B confidence rather than industrializing an
unproven measurement.
