# Cross-Repo Design Handoff Index

**Written:** 2026-06-08  
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
resolution. None of the ADRs are merged or have open PRs yet; all are status
"Proposed."

---

## 3. Design decisions (ADRs)

| ADR | Decision (one line) | Branch | Origin SHA | Status |
|-----|---------------------|--------|------------|--------|
| [ADR-001](ADRs/proposed/ADR-001-referential-axis-alignment.md) | Add `analyze_axis_alignment` tool: project sentences onto an anchor-defined axis, z-scored against a corpus null | `feat/axis-alignment` | `fce5abe` | Proposed |
| [ADR-002](ADRs/proposed/ADR-002-unified-embedding-adapter.md) | One `EmbeddingAdapter` ABC shared by sk-mcp and thought-vault; `BulkEmbedder` wraps any adapter for corpus-scale work | `docs/adr-002-unified-embedding-adapter` | `ceacd3a` | Proposed |
| [ADR-003](ADRs/proposed/ADR-003-stateless-mcp-contract.md) | sk-mcp MCP tools become stateless (model selection per-call); `model_load`/`model_unload` removed; lifecycle moves to llauncher | `docs/adr-003-stateless-mcp-contract` | `3a9000e` | Proposed |

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
`pip install -e`. The `BulkEmbedder` wrapper (already built on
`feat/embedding-engine`) adds checkpoint/resume, sub-chunking, token-aware
batching, and backoff retries to any adapter. The vault's `EmbeddingBridge`
becomes a deprecated shim over `get_adapter() + BulkEmbedder`. Critically,
`model_name` becomes the underlying model identity (not a backend-prefixed
label), which allows vault-produced embeddings to key into ADR-001's null cache.

Open questions left by ADR-002: exact canonical `model_name` format and how
transport metadata is carried alongside it; whether `BulkEmbedder` lives beside
the adapters or in its own module; migration ordering (shared adapter → vault
cutover → null rebuild); dtype/precision unification; ADR numbering across the
two repos.

### ADR-003 — Stateless MCP control-plane

Removes server-side model state from sk-mcp. Each MCP tool call receives model
selection (`backend`/`model`/`base_url`, env fallback) and resolves a fresh
adapter; no adapter is retained between calls; the cross-call embedding cache
is dropped. `model_load` and `model_unload` are removed; process lifecycle moves
entirely to llauncher. The Gradio UI must own its own client-side cache for
reactive slider behavior. The nv_embed path (SentenceTransformers/PyTorch, not
GGUF) requires llauncher to add a vLLM/SentenceTransformers server type before
sk-mcp can route it through llauncher — tracked in llauncher #155.

Open questions left by ADR-003: whether `StateManager` is fully removed or
retained as a thin stateless resolver; exact per-call parameter shape (routed
`model_id` vs. explicit `backend + base_url + model`); sequencing of llauncher
vLLM/SentenceTransformers expansion relative to sk-mcp's cutover.

---

## 4. Code in flight — BulkEmbedder engine

Branch: `feat/embedding-engine` | Origin SHA: `acaba76`

Three files added in a single commit:

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

Status: built and tested with the `FakeAdapter`; **not yet run against the real
corpus**. The real validation run (sk-mcp #3) is the next concrete step.

---

## 5. Tracked open work (issues)

| Issue | Title | What it unblocks |
|-------|-------|------------------|
| [sk-mcp #2](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/2) | Implement ADR-003: stateless MCP control-plane | sk-mcp MCP surface becomes reproducible and self-contained; prerequisite for clean multi-session use |
| [sk-mcp #3](https://github.com/shanevcantwell/semantic-kinematics-mcp/issues/3) | Run chat-log corpus through BulkEmbedder; validate ~45 min throughput | Validates BulkEmbedder on real data; produces embeddings that serve as ADR-001 axis-alignment null |
| [thought-vault #28](https://github.com/shanevcantwell/thought-vault-integration/issues/28) | Re-extract corpus from richer sources and re-embed | Richer signal (full HTML/markdown, per-message granularity); clean embedding base for all downstream analysis |
| [thought-vault #29](https://github.com/shanevcantwell/thought-vault-integration/issues/29) | Reproducibility: capture embedding-server config; consolidate bulk runner | Documents llauncher `extra_args` (`--embeddings --log-disable`, ubatch/batch 4096) in-repo; decides fate of uncommitted supervisor script |
| [llauncher #155](https://github.com/shanevcantwell/llauncher/issues/155) | Add vLLM/SentenceTransformers server type for non-GGUF embedding models | Lets llauncher own nv_embed (4096-d) lifecycle; prerequisite for ADR-003's full stateless nv_embed path |

---

## 6. Open questions / unresolved

- **ADR numbering scheme.** The ADRs use project-local `ADR-001/002/003` but an
  `adr-namer-draft.sh` script suggests a cross-repo `ADR-CORE-NNN` scheme.
  Which wins, and does it affect the thought-vault's own `ADR-003/004` numbering?

- **Canonical `model_name` format.** ADR-002 Decision 2 says it should be the
  underlying model identity (e.g. `embeddinggemma-300M-F32`), not a
  `Backend:` prefix. The exact format and how transport metadata travels
  alongside it is not yet specified. Changing this string is a
  null-cache-invalidating event.

- **Per-call parameter shape for stateless tools (ADR-003).** Should tools
  accept a single routed `model_id` that the server maps to a backend, or
  explicit `backend + base_url + model` fields? Affects the MCP schema for all
  six analysis tools.

- **`StateManager` fate (ADR-003).** Fully removed, or kept as a thin stateless
  resolver that constructs an adapter from per-call args and holds nothing between
  calls? Unclear which is cleaner given the current dispatch pattern in
  `server.py`.

- **Which readout leads `analyze_axis_alignment` (ADR-001).** Three candidates:
  position trace z-scores, axis drift (net signed march), or axis-restricted
  straightness ratio. Not decided; depends on real-corpus validation.

- **Null-cache staleness beyond model-name match (ADR-001).** Current guard
  refuses a null whose `model_name` differs from the active adapter. Whether to
  add a date/hash-based staleness signal is deferred.

- **Normalization audit (ADR-002).** The LM Studio adapter currently does not
  L2-normalize; embeddinggemma via llama-server behavior is not yet verified. ADR-002
  requires each adapter to declare its normalization contract explicitly; the
  audit has not been done.

- **Sequencing: llauncher #155 vs. sk-mcp ADR-003 cutover.** sk-mcp can go
  stateless against llama-server backends immediately; the nv_embed (PyTorch)
  path through llauncher requires #155 first. Is the partial cutover (llama-server
  stateless now, nv_embed follows) acceptable, or should both wait?

- **thought-vault #29: supervisor script + packing-ceiling edit.** An uncommitted
  `scripts/run_embeddings_supervised.sh` and a bridge throughput edit (packing
  ceiling 1500) on branch `fix/embedding-checkpoint-resume` are likely superseded
  by BulkEmbedder but have not been formally closed or discarded. PR #27
  (checkpoint-resume fix) is still open.

- **BulkEmbedder module location (ADR-002).** Currently
  `semantic_kinematics/embeddings/bulk.py`. ADR-002 notes the question of whether
  it stays beside the adapters or moves to its own module. Not resolved.

---

## 7. Suggested next actions (ordered)

1. **Run the corpus through BulkEmbedder** (sk-mcp #3). This is the highest-value
   next step: it validates the engine on real data, produces embeddings for the
   ADR-001 null, and surfaces any tuning needed before ADR-002 migration work
   starts. Configure llauncher's embeddinggemma with `--embeddings --log-disable`
   and batch/ubatch 4096 (per thought-vault #29) before running.

2. **Capture and commit embedding-server config** (thought-vault #29). Before the
   run, document the required llauncher `extra_args` in the repo so the run is
   reproducible. Decide the fate of the supervisor script and the #27 PR.

3. **Resolve the `model_name` format (ADR-002).** Pick the canonical identity
   string before any corpus embeddings are written — changing it later requires
   rebuilding the null cache and all checkpoint files.

4. **Open PRs for the three ADRs.** They are "Proposed" with no PR yet. Opening
   PRs (even with no merge planned immediately) gives a visible review surface
   and a natural place to resolve ADR-level open questions.

5. **Implement ADR-003** (sk-mcp #2). Rework `server.py` dispatch and tool input
   schemas; resolve `StateManager` fate; delete `commands/model.py`; move UI
   caching client-side. The llama-server backends can go stateless now; the
   nv_embed path can follow once llauncher #155 lands.

6. **llauncher #155: vLLM/SentenceTransformers server type.** Prerequisite for
   sk-mcp's nv_embed path to become fully stateless. Can be parallelized with
   step 5.

7. **ADR-002 migration.** After the `model_name` format is settled and #3 confirms
   BulkEmbedder works on real data: generalize `LMStudioAdapter` to cover
   llama-server; audit normalization per backend; wire `EmbeddingBridge` as a
   deprecated shim; cut the vault over to the shared adapter.

8. **Re-extract and re-embed corpus** (thought-vault #28). Once the shared adapter
   and BulkEmbedder are stable, do the full re-extraction from richer sources
   (full HTML for Gemini, fuller markdown for Claude) and re-embed at
   per-message granularity.

9. **Build the ADR-001 axis-alignment null from the re-embedded corpus.** Once the
   vault corpus is embedded through the shared adapter with a canonical
   `model_name`, run `scripts/build_axis_null.py` to produce the empirical null
   and validate the `analyze_axis_alignment` tool against real signal (absurdist
   books should flare at high sigma relative to the conversation null).
