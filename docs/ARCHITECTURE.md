# Architecture

This document defines the layering invariant of `semantic-kinematics-mcp` (sk-mcp) and makes the boundary between layers enforceable. Its purpose is to distinguish a valid composition from a violation — not to describe what the code currently does, but to state what it must do and to name the gaps that remain.

---

## The invariant (read this first)

Four rules. All four apply simultaneously.

1. **One stateless execution core.** Same inputs produce same outputs. The core holds no session state; model-server lifecycle is delegated out of the core to llauncher (ADR-SKM-0009). A call that returns different results depending on prior calls is a contract breach.

2. **MCP is the sole contract.** The core is reachable through exactly one door: the contracted MCP tool surface, JSON-RPC over stdio. There is no second door.

3. **Consumers orchestrate; they do not extend.** The two consumers — (a) the Gradio UI and (b) agentic tools / MCP clients — sit above the contract. They compose and sequence calls to contracted primitives. They exercise no novel pathways in the core. Every capability a consumer uses already exists as a contracted MCP primitive.

4. **Reaching across layers is an instant fail.** A consumer that bypasses the contract to touch core internals — or code added to the core that exists solely to serve one consumer — is an architectural violation. Not a shortcut. Not a pragmatic compromise. A violation.

---

## The layers

### Control-plane core (stateless execution)

**What lives here:** The analysis primitives — the contracted MCP tool implementations.

- `semantic_kinematics/mcp/commands/` — six command modules: `embeddings.py`, `trajectory.py`, `classification.py`, `axis_alignment.py`, `model.py`, `direction.py`

**Rules:**
- No session state retained between calls. An adapter is resolved, used, and released per call (target of ADR-SKM-0009; not yet implemented).
- No awareness of which consumer is calling. The core cannot contain logic paths that exist only because the Gradio UI needs them.
- Model-server lifecycle is not owned here. Starting and stopping embedding servers is llauncher's responsibility (ADR-SKM-0009).
- MCP is the sole door into this layer. The invariant — "MCP is the sole contract" — scopes precisely to this control-plane core. A consumer that bypasses the MCP contract to reach `mcp/commands/` directly is an instant fail.

### Shared substrate: the embedding adapter

**What lives here:** The `EmbeddingAdapter` abstraction and its backends.

- `semantic_kinematics/embeddings/` — the `EmbeddingAdapter` ABC (`base.py`) and three backends (`nv_embed_adapter.py`, `lmstudio.py`, `sentence_transformers_adapter.py`); the unified adapter target of ADR-SKM-0008

This is a layer **beneath** both the control-plane core and the data-plane. It is not itself the contracted core. The MCP-sole-contract invariant governs access to the control-plane core (`mcp/commands/`), not access to this substrate. Both the analysis core and bulk data-plane jobs rest on the substrate; neither relationship is a bypass of the other.

**Rules:**
- The adapter substrate is the lowest sk-mcp layer. Nothing in it imports from `mcp/commands/` or `mcp/server.py`.
- The control-plane core uses it to execute analysis. The data plane uses it for bulk embedding. Those are different applications of the same substrate.

### Contract (MCP tool surface)

**What lives here:** `semantic_kinematics/mcp/server.py`.

The server dispatches JSON-RPC tool calls to the command modules. This is the sole authorized point of entry into the control-plane core. The contracted tool surface is currently ten tools: `embed_text`, `calculate_drift`, `classify_document`, `analyze_trajectory`, `compare_trajectories`, `analyze_axis_alignment`, `model_status`, `model_load`, `model_unload`, `initialize_direction`.

**Rules:**
- All control-plane consumer access to analysis primitives passes through this surface.
- The contract is versioned; adding or removing a tool is a surface change, not an internal refactor.

### Orchestration / consumer plane

**What lives here:** The Gradio UI (`semantic_kinematics/ui/`) and any external MCP client (Claude Code, scripts, agents).

**Rules:**
- Control-plane consumers compose and sequence contracted MCP tool calls.
- A consumer may cache results client-side for its own reactivity (e.g. the UI slider re-using cached embeddings — see ADR-SKM-0009). That cache lives in the consumer, not the core.
- A control-plane consumer may not import from `semantic_kinematics.mcp.commands.*` directly — that bypasses the contract. The Gradio UI must not import from `semantic_kinematics.embeddings.*` to perform analysis either — that substitutes a direct adapter call for an MCP-contracted analysis call, which also bypasses the contract.
- `BulkEmbedder` is not a control-plane consumer. It is a data-plane application on the shared substrate (ADR-SKM-0009 control-plane/data-plane split). The consumer-plane rules do not classify it as a violator. If a data-plane job needs analysis results, it must re-enter through the MCP contract like any other consumer.

### Lifecycle plane (out of process)

**What lives here:** llauncher, managing embedding-model server processes.

llauncher owns the start/stop/swap/status lifecycle of model servers over its own stateless MCP interface. sk-mcp tools target an already-running endpoint; they do not start, stop, or monitor model servers.

Current scope: llauncher manages llama-server (GGUF) processes. The `nv_embed` path (SentenceTransformers/PyTorch) requires llauncher to add a vLLM/SentenceTransformers server type before sk-mcp can route it through llauncher — tracked as llauncher issue #155.

---

## Diagram

```
┌──────────────────────────────────┐          ┌──────────────────────────────┐
│   ORCHESTRATION / CONSUMER PLANE │          │  DATA PLANE                  │
│                                  │          │                              │
│  ┌──────────────┐  ┌───────────┐ │          │  ┌────────────────────────┐  │
│  │  Gradio UI   │  │  MCP      │ │          │  │  BulkEmbedder          │  │
│  │  (ui/)       │  │  clients/ │ │          │  │  (embeddings/bulk.py)  │  │
│  │              │  │  agents   │ │          │  │                        │  │
│  └──────┬───────┘  └─────┬─────┘ │          │  └──────────┬─────────────┘  │
│         │ MCP tool calls │       │          │             │ adapter calls   │
│         │ (JSON-RPC/     │       │          │             │ only; no        │
│         │  stdio) only   │       │          │             │ mcp/commands/   │
└─────────┼────────────────┼───────┘          └─────────────┼────────────────┘
          │                │                                │
══════════╪════════════════╪════════════════════════════════│═════════════════
          │  CONTRACT BOUNDARY (governs access to analysis  │
══════════╪════════════════╪══════════════════  core only)  │═════════════════
          │                │                                │
          └───────┬────────┘                                │
                  ▼                                         │
  ┌───────────────────────────────┐                         │
  │         MCP SERVER            │                         │
  │  mcp/server.py                │                         │
  │  (sole entry point to core)   │                         │
  └───────────────┬───────────────┘                         │
                  │                                         │
                  ▼                                         │
  ┌───────────────────────────────┐                         │
  │  CONTROL-PLANE CORE           │                         │
  │  (stateless analysis target)  │                         │
  │                               │                         │
  │  mcp/commands/                │                         │
  │    embeddings.py              │                         │
  │    trajectory.py              │                         │
  │    classification.py          │                         │
  │    axis_alignment.py          │                         │
  │    model.py                   │                         │
  └───────────────┬───────────────┘                         │
                  │ uses substrate                          │ uses substrate
                  │                                         │
══════════════════╪═════════════════════════════════════════╪═════════════════
                  │     SHARED SUBSTRATE                    │
══════════════════╪═════════════════════════════════════════╪═════════════════
                  │                                         │
                  └───────────────┬─────────────────────────┘
                                  ▼
                  ┌───────────────────────────────┐
                  │  EMBEDDING ADAPTER            │
                  │  embeddings/                  │
                  │    base.py ← EmbeddingAdapter │
                  │    nv_embed_adapter.py        │
                  │    lmstudio.py                │
                  │    sentence_transformers_      │
                  │      adapter.py               │
                  └───────────────┬───────────────┘
                                  │ adapter calls
                                  ▼
                  ┌───────────────────────────────┐      ┌──────────────────┐
                  │  EMBEDDING BACKENDS (external) │      │  llauncher       │
                  │                               │      │  (out of process)│
                  │  llama-server  (GGUF/OpenAI)  │◄─────│                  │
                  │  NV-Embed-v2  (SentenceXformrs)│      │  start/stop/swap │
                  │  LM Studio    (OpenAI API)    │      │  model servers   │
                  └───────────────────────────────┘      └──────────────────┘
```

Notes on the diagram:
- The contract boundary governs access to the **control-plane core** (`mcp/commands/`) only. Every control-plane consumer arrow crosses it through `server.py`. No lateral arrows run from the consumer plane directly into `mcp/commands/`.
- `BulkEmbedder` is in the data plane — outside and beside the contract boundary. It draws directly on the shared adapter substrate. It does not cross the contract boundary and does not touch `mcp/commands/`.
- Both the control-plane core and `BulkEmbedder` sit on top of the shared adapter substrate. That is a scoping fact, not an exception to the invariant.
- llauncher manages server lifecycle; sk-mcp talks to the resulting running endpoint.

---

## Why stateless

ADR-SKM-0009 requires stateless tools for three reasons.

First, **determinism**: identical inputs plus model selection must yield identical outputs regardless of call history. A stateful adapter makes `embed_text("foo")` depend on which `model_load` ran earlier, turning analysis results into session artifacts rather than reproducible measurements.

Second, **externalized lifecycle**: a server that holds a live adapter owns a resource with a lifecycle — GPU memory, a network connection, a loaded model. That lifecycle is llauncher's responsibility. The moment sk-mcp holds it, the separation breaks and sk-mcp becomes a process manager.

Third, **composability across consumers**: a stateless core can be called by any consumer in any order without one consumer's session state contaminating another's results. This is the precondition for the Gradio UI and agentic tools to co-exist against the same core without interference.

---

## The unified embedding adapter (ADR-SKM-0008)

The `EmbeddingAdapter` ABC (`embeddings/base.py`) is the single interface used to reach any embedding backend. Three concrete adapters exist today: `nv_embed`, `lmstudio`, and `sentence_transformers`. ADR-SKM-0008 generalizes the OpenAI-compatible adapter to cover llama-server and LM Studio through one parameterized implementation.

The adapter layer is a **shared substrate** beneath both the control-plane core and the data plane — not itself a contracted core, and not subject to the MCP-sole-contract rule. That rule scopes to `mcp/commands/` (the analysis primitives). The adapter substrate is the foundation on which those primitives and bulk data-plane jobs both rest.

`BulkEmbedder` (ADR-SKM-0008, `embeddings/bulk.py` on `feat/embedding-engine`) wraps any `EmbeddingAdapter` and adds checkpoint/resume, sub-chunking with vector averaging, token-aware batching, and backoff retries. It is a data-plane application on the shared substrate: it invokes no analysis primitive from `mcp/commands/` and crosses no contract boundary. ADR-SKM-0009's control-plane/data-plane split is the source of this scoping — it defines what the control-plane invariant covers and what lies outside its scope. If a bulk job needed analysis results (e.g. drift scoring on embedded chunks), it would re-enter through the MCP contract like any other consumer.

`model_name` is a canonical model identity, not a backend label (ADR-SKM-0008, Decision 2). The axis-alignment null cache is keyed by `model_name`; a mismatch causes an explicit refusal. This only works if the same underlying model served through different transports produces the same `model_name` string — the unified adapter is where that contract is enforced.

---

## Current conformance (honest)

The invariant above is the target. The code does not yet conform on two points.

| Violation | Why it breaks the invariant | Resolved by |
|-----------|----------------------------|-------------|
| **Cross-layer import (UI → core).** `semantic_kinematics/ui/tabs/drift/handlers.py` imports `calculate_drift` from `semantic_kinematics.mcp.commands.embeddings` directly. `semantic_kinematics/ui/tabs/trajectory/handlers.py` imports `TrajectoryAnalyzer`, `TrajectoryMetrics`, `analyze_trajectory`, `compare_trajectories_handler` from `semantic_kinematics.mcp.commands.trajectory`, and `model_status`, `model_load`, `model_unload` from `semantic_kinematics.mcp.commands.model`. These are in-process Python calls, not MCP tool calls. The contract boundary does not exist in the current code. | There are two doors into the core — MCP JSON-RPC and direct Python import — violating Rule 2 ("MCP is the sole contract") and Rule 3 ("consumers orchestrate through contracted calls only"). | ADR-SKM-0008 (unified adapter consumed at the MCP layer) and ADR-SKM-0009 (stateless control-plane removes the need for the UI to manage model lifecycle directly). The UI must migrate to issuing MCP tool calls and own its embedding cache client-side. |
| **Shared mutable state (StateManager singleton).** `semantic_kinematics/ui/state.py` imports `StateManager` from `semantic_kinematics.mcp.state_manager` and instantiates a process-global singleton (`state_manager = StateManager()`). This singleton holds a live adapter (`_adapter`), backend selection (`_backend`, `_backend_kwargs`), and a cross-call embedding cache (`_embedding_cache`). The UI hands this singleton to core command functions as a first argument, sharing the same mutable state object across both the MCP server's process and the UI's process-local calls. | The core holds session state — the exact condition ADR-SKM-0009 identifies as the statefulness violation. The embedding cache is a cross-call artifact; the live adapter is a retained resource. Rule 1 ("same inputs produce same outputs") is violated whenever cache contents or adapter state diverge between sessions. | ADR-SKM-0009: `StateManager` becomes a stateless resolver (or is removed); the cross-call embedding cache is dropped from the core; the UI owns its own cache for reactive slider behavior; model lifecycle moves to llauncher. |

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
| ADR-SKM-0007: Referential axis-alignment | Adds `analyze_axis_alignment` tool; establishes the empirical null / z-score pattern; defines `model_name`-keyed null cache | `docs/ADRs/proposed/ADR-SKM-0007-referential-axis-alignment.md` |
| ADR-SKM-0008: Unified embedding adapter | One `EmbeddingAdapter` ABC for all backends; `BulkEmbedder` for batch corpus work; canonical `model_name` as model identity (not backend label); closes the normalization inconsistency across backends | `docs/ADRs/proposed/ADR-SKM-0008-unified-embedding-adapter.md` |
| ADR-SKM-0009: Stateless MCP control-plane | Removes `StateManager` statefulness; makes each tool call self-contained with per-call model selection; removes `model_load`/`model_unload`; delegates all server lifecycle to llauncher; forces the UI to own its cache client-side — the ADR that directly closes Violation 1 and Violation 2 above | `docs/ADRs/proposed/ADR-SKM-0009-stateless-mcp-contract.md` |

All three ADRs are status **Proposed**. None has been implemented. The stateless target described in this document is partly aspirational; the current conformance section above is the accurate description of where the code stands today.

---

## Data pipeline (bulk embedding)

sk-mcp's analysis tools operate on vectors. Producing those vectors at corpus scale is the job of the **data plane**, which sits beside the contract boundary (see the diagram) and rests directly on the shared adapter substrate.

```
chat logs ──▶ [sibling: thought-vault-integration] ──▶ chunked JSONL ──▶ BulkEmbedder ──▶ vectors (checkpoint JSONL)
              adapters → UnifiedMessage → exporter        (corpus)        (this repo)        │
                                                                                            ├─▶ axis-alignment background null  (scripts/build_axis_null.py)
                                                                                            └─▶ downstream analysis (re-entered via the MCP contract)
```

- **Upstream (sibling repo):** raw conversation logs are parsed by `thought-vault-integration`'s adapters (`claude_export`, `pi_session`, `gemini_*`, …) into a `UnifiedMessage` stream, then exported to a chunked JSONL corpus. That repo owns ingestion; sk-mcp owns embedding and analysis. The boundary is a JSONL corpus file.
- **BulkEmbedder (`embeddings/bulk.py`):** wraps any `EmbeddingAdapter`; adds windowed, self-describing checkpoint/resume (a sidecar `<checkpoint>.meta.json` records `model_name` + `dimensions`, and resume fails loud on a model/dim mismatch rather than merging incompatible vectors — #16), token-aware batching, sub-chunking with vector averaging, and backoff retries. It streams prep + embed in windows of `prep_window` items (default 256) rather than prepping the whole corpus up front, so a restart re-preps only the not-yet-checkpointed remainder — the entire run (prep included) reconstructs from the checkpoint, not just the embedding step. Driven by `scripts/embed_corpus.py` (see README → *Bulk Embedding a Corpus* for the CLI; backend chosen via `--backend`, not `EMBEDDING_BACKEND`). It is a data-plane application on the shared substrate — it invokes no `mcp/commands/` primitive and crosses no contract boundary. If a bulk job needs analysis results, it re-enters through the MCP contract like any other consumer.
- **Run wrappers:** `scripts/embed_status.py` reports `done failed pending total` for a (corpus, checkpoint) pair — the truthful completion signal, since `embed_corpus.py` exits 0 even when items are marked `_failed`. `scripts/embed_full_corpus.sh` is the recommended way to run a full corpus: a resumable wrapper that auto-restarts on crash, decides completion from the success count (via `embed_status.py`) rather than the process exit code, and defaults to `nv_embed`-appropriate token budgets (8000/8000).
- **Downstream:** the vectors feed the axis-alignment **background null** (z-scores are meaningless without one — `scripts/build_axis_null.py` builds it, keyed by `model_name`) and any other analysis.

---

## Tool reference

Request/response schemas for the nine contracted MCP tools. All tools return errors as `{ "error": "..." }`.

### embed_text

```json
{
  "text": "string (required)",
  "full_vector": "boolean (default: false)",
  "model": "string (optional; currently informational only — the active backend is used regardless. Per-call model selection lands with ADR-SKM-0009)"
}
```

Returns `embedding_preview` (first 10 dimensions) by default; set `full_vector: true` for the complete vector.

### calculate_drift

```json
{ "text_a": "string (required)", "text_b": "string (required)" }
```

Returns `drift` (0.0–1.0+) and `interpretation`:

| Range | Meaning |
|-------|---------|
| 0.0–0.1 | Very similar |
| 0.1–0.3 | Related |
| 0.3–0.5 | Moderate divergence |
| 0.5–0.7 | Different semantics |
| 0.7+ | Unrelated |

### classify_document

```json
{
  "content": "string (required, truncated to 2000 chars)",
  "categories": { "category-a": "Description or exemplar text", "category-b": "..." },
  "threshold": "number (default: 0.85)"
}
```

Returns `best_match`, `similarity`, `confident` (boolean), `all_similarities`.

### analyze_trajectory

```json
{
  "text": "string (required, 2+ sentences)",
  "acceleration_threshold": "number (default: 0.3)",
  "include_sentences": "boolean (default: false)"
}
```

| Field | Description |
|-------|-------------|
| `n_sentences` | Sentence count |
| `mean_velocity` | Average pacing between sentences |
| `velocity_variance` | Pacing consistency |
| `mean_acceleration` | Average rhythm change |
| `max_acceleration` | Largest pacing spike |
| `acceleration_spikes` | Spikes above threshold, with position + isolation score |
| `deadpan_score` | Isolated spikes against calm background (0–1) |
| `heller_score` | Circular structure with deceleration (0–1) |
| `circularity_score` | Semantic looping (sentence i resembles i-2) |
| `tautology_density` | High pairwise similarity + low net displacement |

### compare_trajectories

```json
{
  "golden_text": "string (required)",
  "synthetic_text": "string (required)",
  "acceleration_threshold": "number (default: 0.3)"
}
```

Fitness (lower = closer match): DTW on acceleration profiles, Pearson correlation, spike position/count matching.

| Fitness | Meaning |
|---------|---------|
| < 0.3 | Excellent structural match |
| 0.3–0.5 | Good match, some rhythm deviation |
| 0.5–0.7 | Moderate — structure present but weak |
| > 0.7 | Poor match |

### analyze_axis_alignment

```json
{
  "text": "string (required, 2+ sentences)",
  "anchor_positive": "string (required, newline-separated exemplars, averaged)",
  "anchor_negative": "string (optional; defaults to the background-null mean)",
  "background_ref": "string (path to a null manifest; defaults to env AXIS_NULL_MANIFEST)",
  "min_pole_separation": "number (default: 0.05)",
  "include_sentences": "boolean (default: false)"
}
```

A background null is **required** — z-scores are meaningless without it. Build one per backend with `scripts/build_axis_null.py`.

| Field | Description |
|-------|-------------|
| `position_zscores` | Per-sentence position on the axis, in sigma vs the null |
| `axis_drift` | Net signed march along the axis (`z_last − z_first`) |
| `axis_straightness` | Discipline of the march (1.0 = straight, 0.0 = oscillation) |
| `mean_zscore` | Mean axis position across the passage |
| `pole_separation` | `‖e₊ − e₋‖`; "axis underdetermined" fires if anchors embed too close |
| `null_count` | Background embeddings the null was built from |

### model_status / model_load / model_unload

- `model_status` — backend state (type, model name, dimensions, cache size). No parameters.
- `model_load` — `{ "backend": "nv_embed | lmstudio | sentence_transformers", "options": "object (optional)" }`. *Slated for removal under ADR-SKM-0009.*
- `model_unload` — unload current model, clear cache, free GPU memory. No parameters. *Slated for removal under ADR-SKM-0009.*

---

## Analysis methods (the math)

### Trajectory analysis

Treats text as a particle moving through embedding space. Each sentence is a point; the path between them encodes rhetorical structure.

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| Velocity | `‖e[i+1] - e[i]‖` | Pacing — magnitude of semantic shift between sentences |
| Acceleration | `\|v[i+1] - v[i]\|` | Rhythm — rate of pacing change |
| Curvature | Angular deflection between consecutive displacement vectors | Direction change in full embedding space |

**Spike detection.** A spike fires when `a[i] >= threshold` (default 0.3); each records index, magnitude, isolation score (how much it stands out from neighbors), and position ratio (0.0 = start, 1.0 = end).

**Composite scores.**

```
deadpan = 0.25 × spikiness + 0.35 × mean_isolation + 0.20 × background_stability + 0.20 × contrast
heller  = 0.35 × circularity + 0.40 × tautology_density + 0.25 × deceleration
```

- **Deadpan** (0–1): isolated acceleration spikes against a stable background (few spikes, high isolation, low noise, strong contrast).
- **Heller** (0–1): circular structure with deceleration (high pairwise similarity, low net displacement, negative velocity trend).

**Comparison / fitness** (`compare_trajectories`, lower = better): DTW on acceleration profiles, Pearson correlation of interpolated acceleration, spike position/count matching, weighted toward spike-isolation quality (30%).

**Context-window smoothing.** The UI can average N consecutive sentence embeddings before computing metrics, smoothing filler without re-embedding: `smoothed[i] = mean(e[i], …, e[i+w-1])`. Window 1 = no smoothing (default).

**Known limitation.** Velocity collapses high-D displacement to a scalar (L2 norm), discarding direction; acceleration compounds this. PCA and heatmap visualizations compensate by operating on the full embedding matrix.

### Axis alignment

Trajectory analysis is *reflexive* — how a passage moves relative to itself. In high dimensions (NV-Embed-v2 is 4096-D) independently varying vectors are nearly orthogonal by default, so inter-step angles saturate and curvature carries little signal.

Axis alignment is *referential*: you define a direction with anchor exemplars and project the passage onto that fixed axis. Here high dimensionality flips from liability to asset — the background projection concentrates tightly around its mean, so a genuine sustained march along the axis stands out at high sigma. One projection returns the position trace (z-scored per sentence), the axis drift (net signed march), and the axis-restricted straightness (disciplined line vs oscillation).

Two cautions are built in:

- **Anisotropy.** Embeddings cluster in a narrow cone, so raw dot products are biased. Significance is always a z-score against an empirical null, never an absolute alignment. Omitting `anchor_negative` uses the null mean as the negative pole, de-meaning the cone in the same step.
- **The null is the experiment.** The z-score means "relative to *this* background population." A real-conversation corpus and a literary corpus produce different sigmas for the same passage — choose deliberately.

Full per-function math is in [`axis-alignment.md`](axis-alignment.md); design rationale in [ADR-SKM-0007](ADRs/proposed/ADR-SKM-0007-referential-axis-alignment.md).
