# ADR-003: Stateless MCP control-plane contract for sk-mcp

**Status:** Accepted (2026-06-08)
**Date:** 2026-06-07

> **Note on file path:** This file remains at `docs/ADRs/proposed/ADR-003-stateless-mcp-contract.md`
> and will not be moved. Several documents across this repo and a private sibling repo link to
> this exact path; moving it would break those references. The Status field above is authoritative
> for the acceptance state of this ADR.

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

**Open threads (now resolved — see below):**
- ~~Whether `StateManager` is fully removed or kept as a thin stateless resolver.~~ → Resolved: Option A (thin resolver). See Resolution 2.
- ~~Exact per-call parameter shape — a single routed `model_id` vs. explicit `backend` + `base_url` + `model`.~~ → Resolved. See Resolution 1.
- Sequencing of the llauncher vLLM/SentenceTransformers expansion relative to
  sk-mcp's cutover (sk-mcp can go stateless against llama-server backends first;
  nv_embed via llauncher follows once that server type exists — llauncher #155).

---

## Resolved decisions (2026-06-08)

### Resolution 1 — Per-call parameter shape

Every tool call carries two selection parameters:

- **`model_name`** — canonical model identity, the same string llauncher uses
  (ADR-002 Decision 2), e.g. `embeddinggemma-300M-F32`. This is the string on
  which ADR-001's null cache keys. It is not a backend label; it is the model's
  identity independent of transport.
- **`base_url`** (or equivalent host+port) — coordinates of an already-running
  endpoint. sk-mcp targets a live server (Decision 3 of this ADR) and does not
  start it. The endpoint must be running; llauncher owns that.

Environment fallback: `EMBEDDING_MODEL` and `EMBEDDING_SERVER_URL` (already
referenced in `StateManager`'s `_default_backend_kwargs()`, `state_manager.py:29-35`).
When a tool call omits `model_name` or `base_url`, the resolver falls back to
these env vars.

The vocabulary is **routed-by-name in llauncher's terms plus an explicit
endpoint** — not a backend enum baked into the call identity. For the
llama-server-first phase (current), the adapter is the unified OpenAI-compatible
one (ADR-002 Decision 1), so transport is implied by `base_url`; no separate
`backend` field is part of the per-call identity. When the nv_embed path lands
via llauncher (#155), transport metadata rides beside `model_name` as adapter
construction detail, never as part of the call identity that flows through the
tool schema.

This closes the open question: *"per-call param shape (routed model_id vs
explicit backend+base_url+model)"*.

### Resolution 2 — StateManager disposition: thin stateless resolver (Option A)

Keep the `StateManager` class but strip all retained session state. Specifically,
the following are removed:

- The four stateful fields: `_embedding_cache`, `_adapter`, `_backend`,
  `_backend_kwargs` (`state_manager.py:51-54`).
- The `set_backend()` mutator (`state_manager.py:114-127`).
- The cache methods: `get_cached_embedding`, `cache_embedding`, `clear_cache`
  (`state_manager.py:60-74`).

After the cutover, `get_adapter(model_name, base_url)` and
`get_embed_fn(model_name, base_url)` take explicit per-call arguments with env
fallback, construct an adapter, return it or a callable derived from it, and
retain nothing across calls. The class becomes a single seam where per-call
resolution, env-fallback, ADR-002's normalization contract (Decision 4), and
`model_name` canonicalization (Decision 2) all live.

Rationale for Option A over Option B (full removal, replacing `StateManager`
with a free `resolve()` function): the behavioral outcome is identical — equally
stateless — but Option A is a smaller diff. The class boundary is not the
problem; the retained fields are. Removing the fields and mutator is the minimal
correct change; replacing the class with a free function buys nothing and
requires touching every call site.

This closes the open question: *"StateManager removal vs thin resolver"*.

### Exposed defect — `embed_text`'s decorative `model` argument

`commands/embeddings.py` (lines 37-42) advertises a `model` parameter in
`embed_text`'s `inputSchema`. The implementation reads it at line 78
(`model = args.get("model", "nomic-embed-text-v1.5")`) but then immediately
ignores it. The comment at lines 84-85 is explicit:

```python
# Note: model parameter is informational only
# All tools use the configured backend (default: NV-Embed-v2)
```

The `model` value is echoed into the response at line 91 (`"model": model`)
but has no effect on which model actually runs. This is the statefulness
violation in miniature: the tool appears to accept model selection but silently
discards it, allowing `StateManager`'s retained `_adapter` to determine the
actual model. As part of the Resolution 2 cutover, the `model` argument becomes
the honored `model_name` selection — the resolver reads it, constructs the
correct adapter, and the response reflects what actually ran.
