# Architecture

This document defines the layering invariant of `semantic-kinematics-mcp` (sk-mcp) and makes the boundary between layers enforceable. Its purpose is to distinguish a valid composition from a violation — not to describe what the code currently does, but to state what it must do and to name the gaps that remain.

---

## The invariant (read this first)

Four rules. All four apply simultaneously.

1. **One stateless execution core.** Same inputs produce same outputs. The core holds no session state; model-server lifecycle is delegated out of the core to llauncher (ADR-003). A call that returns different results depending on prior calls is a contract breach.

2. **MCP is the sole contract.** The core is reachable through exactly one door: the contracted MCP tool surface, JSON-RPC over stdio. There is no second door.

3. **Consumers orchestrate; they do not extend.** The two consumers — (a) the Gradio UI and (b) agentic tools / MCP clients — sit above the contract. They compose and sequence calls to contracted primitives. They exercise no novel pathways in the core. Every capability a consumer uses already exists as a contracted MCP primitive.

4. **Reaching across layers is an instant fail.** A consumer that bypasses the contract to touch core internals — or code added to the core that exists solely to serve one consumer — is an architectural violation. Not a shortcut. Not a pragmatic compromise. A violation.

---

## The layers

### Core (stateless execution)

**What lives here:** The analysis and embedding primitives.

- `semantic_kinematics/mcp/commands/` — embeddings, trajectory, classification, axis\_alignment, model (six contracted tools)
- `semantic_kinematics/embeddings/` — the `EmbeddingAdapter` ABC and three backends (`nv_embed`, `lmstudio`, `sentence_transformers`); the unified adapter target of ADR-002

**Rules:**
- No session state retained between calls. An adapter is resolved, used, and released per call (target of ADR-003; not yet implemented).
- No awareness of which consumer is calling. The core cannot contain logic paths that exist only because the Gradio UI needs them.
- Model-server lifecycle is not owned here. Starting and stopping embedding servers is llauncher's responsibility (ADR-003).

### Contract (MCP tool surface)

**What lives here:** `semantic_kinematics/mcp/server.py`.

The server dispatches JSON-RPC tool calls to the command modules. This is the sole authorized point of entry into the core. The contracted tool surface is currently nine tools: `embed_text`, `calculate_drift`, `classify_document`, `analyze_trajectory`, `compare_trajectories`, `analyze_axis_alignment`, `model_status`, `model_load`, `model_unload`.

**Rules:**
- All consumer access to core primitives passes through this surface.
- The contract is versioned; adding or removing a tool is a surface change, not an internal refactor.

### Orchestration / consumer plane

**What lives here:** The Gradio UI (`semantic_kinematics/ui/`) and any external MCP client (Claude Code, scripts, agents).

**Rules:**
- Consumers compose and sequence contracted MCP tool calls.
- A consumer may cache results client-side for its own reactivity (e.g. the UI slider re-using cached embeddings — see ADR-003). That cache lives in the consumer, not the core.
- A consumer may not import from `semantic_kinematics.mcp.commands.*` or `semantic_kinematics.embeddings.*` directly. Those are core internals.

### Lifecycle plane (out of process)

**What lives here:** llauncher, managing embedding-model server processes.

llauncher owns the start/stop/swap/status lifecycle of model servers over its own stateless MCP interface. sk-mcp tools target an already-running endpoint; they do not start, stop, or monitor model servers.

Current scope: llauncher manages llama-server (GGUF) processes. The `nv_embed` path (SentenceTransformers/PyTorch) requires llauncher to add a vLLM/SentenceTransformers server type before sk-mcp can route it through llauncher — tracked as llauncher issue #155.

---

## Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION / CONSUMER PLANE                │
│                                                                 │
│   ┌──────────────────┐          ┌──────────────────────────┐   │
│   │   Gradio UI      │          │  MCP clients / agents    │   │
│   │  (semantic_      │          │  (Claude Code, scripts)  │   │
│   │   kinematics/ui) │          │                          │   │
│   └────────┬─────────┘          └────────────┬─────────────┘   │
│            │  MCP tool calls only             │                 │
│            │  (JSON-RPC over stdio)           │                 │
└────────────┼─────────────────────────────────┼─────────────────┘
             │                                  │
═════════════╪══════════════════════════════════╪═════════════════
             │         CONTRACT BOUNDARY        │
═════════════╪══════════════════════════════════╪═════════════════
             │                                  │
             └──────────────┬───────────────────┘
                            ▼
             ┌──────────────────────────────────┐
             │         MCP SERVER               │
             │  semantic_kinematics/mcp/         │
             │  server.py  (sole entry point)   │
             └──────────────────────────────────┘
                            │
                            ▼
             ┌──────────────────────────────────┐
             │      CORE (stateless target)     │
             │                                  │
             │  mcp/commands/                   │
             │    embeddings.py                 │
             │    trajectory.py                 │
             │    classification.py             │
             │    axis_alignment.py             │
             │    model.py                      │
             │                                  │
             │  embeddings/  (adapter layer)    │
             │    base.py  ← EmbeddingAdapter   │
             │    nv_embed_adapter.py           │
             │    lmstudio.py                   │
             │    sentence_transformers_adapter  │
             │    [bulk.py — ADR-002 BulkEmbedder│
             │     for any batch corpus]        │
             └────────────┬─────────────────────┘
                          │ adapter calls
                          ▼
             ┌──────────────────────────────────┐      ┌──────────────────┐
             │   EMBEDDING BACKENDS (external)  │      │  llauncher       │
             │                                  │      │  (out of process)│
             │   llama-server  (GGUF/OpenAI)    │◄─────│                  │
             │   NV-Embed-v2   (SentenceXformrs)│      │  start/stop/swap │
             │   LM Studio     (OpenAI API)     │      │  model servers   │
             └──────────────────────────────────┘      └──────────────────┘
```

Notes on the diagram:
- Every consumer arrow crosses the contract boundary through `server.py`. There are no lateral arrows from the consumer plane directly into `mcp/commands/` or `embeddings/`.
- `BulkEmbedder` (ADR-002) is a batch consumer of the adapter layer — it does not add a new path through the contract; bulk embedding jobs use the adapter directly without going through the MCP server.
- llauncher manages server lifecycle; sk-mcp talks to the resulting running endpoint.

---

## Why stateless

ADR-003 requires stateless tools for three reasons.

First, **determinism**: identical inputs plus model selection must yield identical outputs regardless of call history. A stateful adapter makes `embed_text("foo")` depend on which `model_load` ran earlier, turning analysis results into session artifacts rather than reproducible measurements.

Second, **externalized lifecycle**: a server that holds a live adapter owns a resource with a lifecycle — GPU memory, a network connection, a loaded model. That lifecycle is llauncher's responsibility. The moment sk-mcp holds it, the separation breaks and sk-mcp becomes a process manager.

Third, **composability across consumers**: a stateless core can be called by any consumer in any order without one consumer's session state contaminating another's results. This is the precondition for the Gradio UI and agentic tools to co-exist against the same core without interference.

---

## The unified embedding adapter (ADR-002)

The `EmbeddingAdapter` ABC (`embeddings/base.py`) is the single interface the core uses to reach any embedding backend. Three concrete adapters exist today: `nv_embed`, `lmstudio`, and `sentence_transformers`. ADR-002 generalizes the OpenAI-compatible adapter to cover llama-server and LM Studio through one parameterized implementation.

`BulkEmbedder` (ADR-002, `embeddings/bulk.py` on `feat/embedding-engine`) wraps any `EmbeddingAdapter` and adds checkpoint/resume, sub-chunking with vector averaging, token-aware batching, and backoff retries. It is a batch consumer of the adapter — not a parallel pathway into the core. A bulk embedding job (e.g. embedding a large background corpus for axis-alignment null construction) calls `BulkEmbedder` directly against the adapter layer; it does not go through the MCP server. This is legitimate: ADR-003 explicitly separates the data-plane (bulk ingestion, which need not be MCP) from the control-plane (analysis tools, which must be stateless MCP calls).

`model_name` is a canonical model identity, not a backend label (ADR-002, Decision 2). The axis-alignment null cache is keyed by `model_name`; a mismatch causes an explicit refusal. This only works if the same underlying model served through different transports produces the same `model_name` string — the unified adapter is where that contract is enforced.

---

## Current conformance (honest)

The invariant above is the target. The code does not yet conform on two points.

| Violation | Why it breaks the invariant | Resolved by |
|-----------|----------------------------|-------------|
| **Cross-layer import (UI → core).** `semantic_kinematics/ui/tabs/drift/handlers.py` imports `calculate_drift` from `semantic_kinematics.mcp.commands.embeddings` directly. `semantic_kinematics/ui/tabs/trajectory/handlers.py` imports `TrajectoryAnalyzer`, `TrajectoryMetrics`, `analyze_trajectory`, `compare_trajectories_handler` from `semantic_kinematics.mcp.commands.trajectory`, and `model_status`, `model_load`, `model_unload` from `semantic_kinematics.mcp.commands.model`. These are in-process Python calls, not MCP tool calls. The contract boundary does not exist in the current code. | There are two doors into the core — MCP JSON-RPC and direct Python import — violating Rule 2 ("MCP is the sole contract") and Rule 3 ("consumers orchestrate through contracted calls only"). | ADR-002 (unified adapter consumed at the MCP layer) and ADR-003 (stateless control-plane removes the need for the UI to manage model lifecycle directly). The UI must migrate to issuing MCP tool calls and own its embedding cache client-side. |
| **Shared mutable state (StateManager singleton).** `semantic_kinematics/ui/state.py` imports `StateManager` from `semantic_kinematics.mcp.state_manager` and instantiates a process-global singleton (`state_manager = StateManager()`). This singleton holds a live adapter (`_adapter`), backend selection (`_backend`, `_backend_kwargs`), and a cross-call embedding cache (`_embedding_cache`). The UI hands this singleton to core command functions as a first argument, sharing the same mutable state object across both the MCP server's process and the UI's process-local calls. | The core holds session state — the exact condition ADR-003 identifies as the statefulness violation. The embedding cache is a cross-call artifact; the live adapter is a retained resource. Rule 1 ("same inputs produce same outputs") is violated whenever cache contents or adapter state diverge between sessions. | ADR-003: `StateManager` becomes a stateless resolver (or is removed); the cross-call embedding cache is dropped from the core; the UI owns its own cache for reactive slider behavior; model lifecycle moves to llauncher. |

---

## What "instant fail" looks like

Concrete examples of violations versus their valid forms:

| Violation | Valid form |
|-----------|------------|
| The UI imports a function from `semantic_kinematics.mcp.commands.*` and calls it directly. | The UI issues a JSON-RPC MCP tool call to `server.py` and receives the result. |
| A code path is added to `mcp/commands/trajectory.py` that exists solely because the Gradio slider needs it (e.g. a `recompute_from_cache` variant that uses the server-side cache). | The UI caches the embeddings client-side after the initial `analyze_trajectory` call; slider updates recompute locally without hitting the core again. |
| A consumer mutates `state_manager._embedding_cache` or `state_manager._adapter` directly. | A consumer is stateless relative to the core; any caching it does lives in its own local data structure. |
| A new tool is added to `mcp/commands/` with no corresponding entry in `server.py`'s dispatch table. | Every tool implemented in `commands/` is registered in `server.py`'s `list_tools()` and dispatched in `call_tool()`. The two are kept in sync; the server is the complete surface. |

---

## Relation to the ADRs

| ADR | What it fixes | File |
|-----|--------------|------|
| ADR-001: Referential axis-alignment | Adds `analyze_axis_alignment` tool; establishes the empirical null / z-score pattern; defines `model_name`-keyed null cache | `docs/ADRs/proposed/ADR-001-referential-axis-alignment.md` |
| ADR-002: Unified embedding adapter | One `EmbeddingAdapter` ABC for all backends; `BulkEmbedder` for batch corpus work; canonical `model_name` as model identity (not backend label); closes the normalization inconsistency across backends | `docs/ADRs/proposed/ADR-002-unified-embedding-adapter.md` |
| ADR-003: Stateless MCP control-plane | Removes `StateManager` statefulness; makes each tool call self-contained with per-call model selection; removes `model_load`/`model_unload`; delegates all server lifecycle to llauncher; forces the UI to own its cache client-side — the ADR that directly closes Violation 1 and Violation 2 above | `docs/ADRs/proposed/ADR-003-stateless-mcp-contract.md` |

All three ADRs are status **Proposed**. None has been implemented. The stateless target described in this document is partly aspirational; the current conformance section above is the accurate description of where the code stands today.
