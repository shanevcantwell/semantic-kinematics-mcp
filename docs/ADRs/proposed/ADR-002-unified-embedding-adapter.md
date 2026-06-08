# ADR-002: Unified embedding adapter across kinematics and thought-vault

**Status:** Proposed
**Date:** 2026-06-07

## Context

Two sibling repos embed text against the same kinds of backends, with two
separately-maintained implementations that have drifted apart:

- **`semantic-kinematics-mcp`** owns a clean abstraction: an `EmbeddingAdapter`
  ABC (`model_name`, `dimensions`, `embed`, `embed_batch`, `unload`,
  `is_loaded`, cosine helpers) and a `get_adapter(backend, **kwargs)` factory
  over three backends — `nv_embed` (SentenceTransformers/NV-Embed-v2, 4096-d,
  L2-normalized), `sentence_transformers` (generic), and `lmstudio`
  (OpenAI-compatible `/v1/embeddings`). It has **no bulk-corpus robustness** —
  one batch call, no resume, no retries, no sub-chunking.

- **`thought-vault-integration`** owns `EmbeddingBridge`: a standalone class
  that POSTs to the **same OpenAI-compatible `/v1/embeddings`** route (a
  llama-server at `inference-host:8082` serving `embeddinggemma-300M-F32`,
  768-d). It is **built for bulk**: checkpoint/resume (now `_failed`-aware, per
  ADR-style note `docs/embedding-checkpoint-bug.md` in that repo), sub-chunking
  of >context-window texts with vector averaging, token-aware batching,
  exponential-backoff retries, and connection pooling. Its model and endpoint
  are **hardcoded**, with no override.

The vault's README states the bridge *"replaces the broken `semantic_kinematics`
import path"* — i.e. the vault **once imported the adapter from kinematics** and
forked a private copy when that import broke. The current split is therefore a
**regression to heal**, not a deliberate boundary.

Three facts make unification more than tidiness:

1. **The OpenAI-compatible path is literally the same protocol.** The kinematics
   `LMStudioAdapter` and the vault `EmbeddingBridge` differ only in HTTP client
   (`openai` vs `requests`) and in the bulk machinery layered on top.
2. **ADR-001's axis-alignment null is keyed by `model_name` and refuses a
   mismatch.** Today the vault produces `embeddinggemma-300M-F32.gguf` while the
   kinematics LM Studio adapter produces `LMStudio:<model>` — so the vault's
   real-conversation corpus, the exact empirical null ADR-001 wants, **cannot be
   dropped into the null cache**. A shared adapter with a canonical `model_name`
   dissolves this by construction.
3. **Normalization is inconsistent and load-bearing.** The SentenceTransformers
   adapter L2-normalizes by default; the LM Studio adapter does **not**; the
   vault bridge **averages sub-chunk vectors**, which breaks unit norm. ADR-001's
   z-score sharpness depends on L2-normalized inputs. A single code path is the
   only place to make normalization a deliberate, consistent contract.

## Decision

1. **One `EmbeddingAdapter` contract, shared.** Keep the kinematics ABC as the
   canonical interface. Generalize the OpenAI-compatible adapter (the current
   `LMStudioAdapter`) into a llama-server/OpenAI-compatible adapter that works
   for LM Studio **and** llama-server, parameterized by `base_url` and `model`.

2. **`model_name` is a canonical model identity, not a backend label.** The null
   match in ADR-001 requires that the *same model served two ways* yields the
   *same* `model_name`. Decision: `model_name` reports the underlying model id
   (e.g. `embeddinggemma-300M-F32`), not a `Backend:` prefix. Backend/transport
   becomes separate metadata, not part of the identity the null keys on.

3. **Bulk robustness is a wrapper, not adapter-internal.** Introduce a
   `BulkEmbedder` that wraps *any* `EmbeddingAdapter` and adds the vault's
   machinery: `_failed`-aware checkpoint/resume, sub-chunk + vector-average for
   long texts, token-aware batching, backoff retries, connection reuse. Adapters
   stay thin; bulk is opt-in. The vault's `embed_queue` becomes a thin call into
   `BulkEmbedder`.

4. **Normalization is an explicit adapter contract.** Each adapter declares
   whether it returns L2-normalized vectors; `BulkEmbedder` re-normalizes after
   sub-chunk averaging when the adapter contract is "normalized." ADR-001
   consumers can then assume unit norm without guessing.

5. **Config, never hardcode.** Model and endpoint resolve from factory
   args → env → documented default, in both repos. The vault loses its hardcode.

6. **Home of the shared code — two candidates, recommendation inside.**
   - **(A) Live in `semantic-kinematics-mcp`; the vault depends on it** via
     `pip install -e ../semantic-kinematics-mcp`. Least new infrastructure —
     kinematics already owns the ABC, factory, and three backends. Heals the
     original broken import directly. The personal-data boundary is **not**
     violated: the adapter carries no personal data; only the vault's *corpus*
     does, and that stays local per ADR-001 Decision 4.
   - **(B) Extract a standalone `embedding-adapters` package** both repos
     depend on. Cleanest separation, but a new repo/package and two more
     dependency edges to version and release.

   **Recommendation: (A).** It restores the relationship that already existed,
   adds no new release surface, and keeps the public fork as the natural home of
   general-purpose (non-personal) embedding machinery. Revisit (B) only if a
   third consumer appears or the dependency direction becomes awkward.

7. **Migration is non-breaking.** `EmbeddingBridge` is retained as a thin
   deprecated shim over `get_adapter(...) + BulkEmbedder`, preserving its public
   `embed_queue(...)` signature so the vault orchestrator is untouched at the
   call site. Removal is a later, separate step.

## Consequences

- **The ADR-001 null gains its intended corpus.** Once the vault embeds through
  the shared adapter, its real-conversation embeddings carry a matching
  `model_name` and drop straight into the axis-alignment null cache — the
  "real-exchange corpus as empirical null" ADR-001 named as the real validation.
- **One embedding code path** to maintain; the vault's bulk robustness becomes
  available to kinematics' own bulk needs (e.g. building large nulls).
- **Costs / risks:** a cross-repo dependency edge (kinematics ← vault); a
  `model_name` identity change is a **null-cache-invalidating** event (existing
  caches keyed by old strings must be rebuilt or remapped); normalization
  semantics must be audited per backend during migration (the LM Studio adapter
  currently does not normalize; embeddinggemma via llama-server must be checked).
- **Deferred:** dtype/precision unification; a model/dimension registry; whether
  normalization belongs in the adapter or strictly downstream; retiring
  `EmbeddingBridge` after the shim bake-in.

**Cross-refs:** ADR-001 (`docs/ADRs/proposed/ADR-001-referential-axis-alignment.md`,
the null `model_name` guard); `embeddings/base.py` (the ABC),
`embeddings/__init__.py` (the factory), `embeddings/lmstudio.py` (the adapter to
generalize), `embeddings/sentence_transformers_adapter.py` (normalization
contract); thought-vault `thought_vault_integration/embedding_bridge.py` (the
bulk machinery to extract) and its `docs/embedding-checkpoint-bug.md`.

**Open threads:**
- ADR numbering across the two repos (kinematics `ADR-00N` vs the vault's own
  `ADR-003/004` scheme).
- Exact canonical-`model_name` format and how transport metadata travels
  alongside it.
- Whether `BulkEmbedder` lives beside the adapters or in its own module.
- Migration ordering: shared adapter first, then vault cutover, then null rebuild.
