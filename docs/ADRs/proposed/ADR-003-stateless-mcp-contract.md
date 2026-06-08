# ADR-003: Stateless MCP control-plane contract for sk-mcp

**Status:** Proposed
**Date:** 2026-06-07

## Context

sk-mcp's current MCP contract is **stateful**, which conflicts with the
project's stated requirement that the MCP surface be the *controls* and that
those controls be **stateless calls**:

- `StateManager` holds authoritative server-side state: a live `_adapter`, the
  chosen `_backend`/`_backend_kwargs`, and a cross-call `_embedding_cache`
  (`state_manager.py:51-54`).
- `model_load` / `model_unload` **mutate** that state; `model_status` reads it
  (`commands/model.py:100-160`).
- Every analysis tool resolves through `manager.get_adapter()`, so
  `embed_text("foo")` returns *different vectors depending on which `model_load`
  ran earlier in the session*. The call is not self-contained — the exact
  property the stateless pattern forbids.

Sibling repos already codify the opposite as explicit doctrine:

- **llauncher ADR-008 "Stateless Facade"** — the canonical statement: the facade
  *"owns no data … nothing is cached at the facade layer; every call queries the
  underlying sources fresh."* Sources of truth are external (config, process
  table, env).
- **prompt-prix** — *"9 stateless tools … model_id passed per-call, the registry
  routes."* Stateful GPU management is externalized to `local-inference-pool`,
  out of the tool layer.
- **frontier-advisor** — *"The server is stateless: it routes and returns."*
  Access policy is the scaffold's job, not the server's.

The shared rule: the MCP server holds no session/loaded-model state, and
**stateful resource/process lifecycle lives outside the tool layer**. llauncher
already owns llama-server process lifecycle via stateless `start/stop/swap/status`.

## Decision

1. **sk-mcp MCP tools are stateless.** Each call is self-contained: the model
   selection (`backend` / `model` / `base_url`, falling back to env) is passed
   per-call; a fresh adapter is resolved, used, and discarded. No server-held
   adapter, no retained backend selection. Applies to `embed_text`,
   `calculate_drift`, `analyze_trajectory`, `compare_trajectories`,
   `classify_document`, `analyze_axis_alignment`.

2. **Control plane vs. data plane.** The MCP surface is the **control plane**
   (stateless). Bulk **data ingestion** (the `BulkEmbedder` of ADR-002) runs
   *outside* MCP, directly against the model endpoint, but is governed by the
   same per-call model selection — "ingestion need not be MCP; the controls must."

3. **Model lifecycle leaves sk-mcp; llauncher owns it.** Remove `model_load` and
   `model_unload`. Starting/stopping model servers is llauncher's stateless MCP
   domain; sk-mcp tools target an already-running endpoint.
   - **Cross-repo dependency:** llauncher today manages llama-server (GGUF)
     processes. NV-Embed-v2 runs via SentenceTransformers/PyTorch (not GGUF).
     For llauncher to own *all* embedding-model lifecycle it must expand to a
     **vLLM and/or SentenceTransformers server type**. Accepted as a direction;
     tracked as an llauncher roadmap item.

4. **No facade-level cache.** Drop `StateManager`'s cross-call embedding cache —
   it is the statefulness violation. Caching for UI reactivity (e.g. trajectory
   sliders) moves to the **client/UI**. (ADR-008's down-pushed-TTL alternative
   was considered and declined for the facade; the client owns reactivity.)

5. **`StateManager` becomes a stateless resolver** (or is removed): construct an
   adapter from per-call args, retain nothing between calls.

## Consequences

- Calls are reproducible and self-contained — identical inputs + model selection
  yield identical outputs regardless of call history. This pays off ADR-001
  (canonical `model_name` per call → null reuse stops being a special case) and
  ADR-002 (one adapter contract, per-call selection).
- **Breaking tool-surface change:** `model_load`/`model_unload` are removed.
- **UI behavioral change:** the Gradio UI must own its own embedding cache for
  reactive sliders, since the facade no longer caches.
- **llauncher roadmap:** non-GGUF (vLLM/SentenceTransformers) server types for
  the nv_embed path.
- **Migration scope:** rework `server.py` dispatch + tool input schemas to accept
  per-call model args; convert/remove `StateManager`; delete `commands/model.py`;
  move UI caching client-side; update tests.

**Cross-refs:** ADR-001 (`docs/ADRs/proposed/ADR-001-referential-axis-alignment.md`,
null `model_name` guard); ADR-002 (`docs/ADRs/proposed/ADR-002-unified-embedding-adapter.md`,
unified adapter + BulkEmbedder); llauncher `docs/adrs/accepted/008-launcher-state-stateless-facade.md`;
prompt-prix / frontier-advisor `ARCHITECTURE.md`; `state_manager.py`;
`commands/model.py`; `server.py`.

**Open threads:**
- Whether `StateManager` is fully removed or kept as a thin stateless resolver.
- Exact per-call parameter shape — a single routed `model_id` vs. explicit
  `backend` + `base_url` + `model`.
- Sequencing of the llauncher vLLM/SentenceTransformers expansion relative to
  sk-mcp's cutover (sk-mcp can go stateless against llama-server backends first;
  nv_embed via llauncher follows once that server type exists).
