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
   real data (~80K messages, ~45 min target) and surfaces tuning needs before
   ADR-002 migration work. Note: until #11 item 2 lands, these embeddings are
   keyed by the current adapter's `model_name` and are validation output, not
   yet null-cache-eligible.

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
