<img width="1920" height="1072" alt="image" src="https://github.com/user-attachments/assets/eeb237a7-2fc6-414d-965b-26a116e8a739" />
Embedding space analysis toolkit. Measures semantic drift between texts, traces trajectory dynamics through prose, projects passages onto caller-defined semantic axes, and exposes everything as MCP tools for agentic integration.

## Quick Start

```bash
# MCP server only (lean install)
pip install -e .

# With Gradio UI
pip install -e ".[ui]"

# With GPU support (NV-Embed-v2, ~14GB VRAM)
pip install -e ".[gpu]"

# UI + GPU
pip install -e ".[ui,gpu]"

# Start MCP server
semantic-kinematics-mcp

# Or launch Gradio UI (requires [ui])
python -m semantic_kinematics
```

### Docker

```bash
docker build -t mcp/semantic-kinematics .
docker run -i --rm mcp/semantic-kinematics
```

Or with docker-compose for host networking and data mounts:

```bash
docker-compose up
```

## Architecture

A single stateless core, reachable only through the MCP contract. The Gradio UI and agentic tools (MCP clients) orchestrate by composing contracted tool calls — they never reach across the contract boundary into core internals. Model-server lifecycle lives outside the core, managed by llauncher.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full layering invariant, layer definitions, ASCII diagram, and current conformance gaps.

## Embedding Backends

Three interchangeable backends, selected via `EMBEDDING_BACKEND` environment variable:

| Backend | Model | Dimensions | Notes |
|---------|-------|------------|-------|
| `nv_embed` | NV-Embed-v2 | 4096 | GPU, fp16, highest quality |
| `lmstudio` | Any GGUF via OpenAI API | Varies | Local LM Studio server |
| `sentence_transformers` | Any HuggingFace model | Varies | General purpose |

**NV-Embed-v2 note:** This model uses a custom `BidirectionalMistralModel` that resists standard quantization (bitsandbytes int8/int4) and GGUF conversion. fp16 (~14GB VRAM) is the practical minimum. For lower VRAM requirements, use `lmstudio` or `sentence_transformers` with a smaller model.

Configure in `.env`:

```
EMBEDDING_BACKEND=nv_embed
```

## MCP Tools

9 tools over JSON-RPC (stdio).

| Tool | Description |
|------|-------------|
| `embed_text` | Get embedding vector for text |
| `calculate_drift` | Cosine distance between two texts |
| `classify_document` | Similarity-based document classification |
| `analyze_trajectory` | Velocity, acceleration, curvature metrics for a passage |
| `compare_trajectories` | Fitness score: compare two passages structurally |
| `analyze_axis_alignment` | Project a passage onto a caller-defined semantic axis, z-scored against a background null |
| `model_status` | Check embedding backend state |
| `model_load` | Load a specific backend |
| `model_unload` | Unload model and free memory |

### Configure in Claude Code

```json
{
  "mcpServers": {
    "semantic-kinematics": {
      "command": "semantic-kinematics-mcp",
      "env": {
        "EMBEDDING_BACKEND": "nv_embed"
      }
    }
  }
}
```

### Tool Reference

#### embed_text

Get embedding vector for text.

```json
{
  "text": "string (required)",
  "full_vector": "boolean (default: false)",
  "model": "string (optional, override backend model)"
}
```

Returns `embedding_preview` (first 10 dimensions) by default. Set `full_vector: true` for the complete vector.

#### calculate_drift

Cosine distance between two texts.

```json
{
  "text_a": "string (required)",
  "text_b": "string (required)"
}
```

Returns `drift` (0.0–1.0+) and `interpretation`:

| Range | Meaning |
|-------|---------|
| 0.0–0.1 | Very similar |
| 0.1–0.3 | Related |
| 0.3–0.5 | Moderate divergence |
| 0.5–0.7 | Different semantics |
| 0.7+ | Unrelated |

#### classify_document

Classify text by cosine similarity to category exemplars.

```json
{
  "content": "string (required, truncated to 2000 chars)",
  "categories": {
    "category-a": "Description or exemplar text for category A",
    "category-b": "Description or exemplar text for category B"
  },
  "threshold": "number (default: 0.85)"
}
```

Returns `best_match`, `similarity`, `confident` (boolean), and `all_similarities`.

#### analyze_trajectory

Compute velocity, acceleration, and curvature for a text passage. Each sentence becomes a point in embedding space; metrics describe the path between them.

```json
{
  "text": "string (required, 2+ sentences)",
  "acceleration_threshold": "number (default: 0.3)",
  "include_sentences": "boolean (default: false)"
}
```

Returns:

| Field | Description |
|-------|-------------|
| `n_sentences` | Sentence count |
| `mean_velocity` | Average pacing between sentences |
| `velocity_variance` | Pacing consistency |
| `mean_acceleration` | Average rhythm change |
| `max_acceleration` | Largest pacing spike |
| `acceleration_spikes` | List of spikes above threshold, with position and isolation score |
| `deadpan_score` | Isolated spikes against calm background (0–1) |
| `heller_score` | Circular structure with deceleration (0–1) |
| `circularity_score` | Semantic looping (sentence i resembles sentence i-2) |
| `tautology_density` | High pairwise similarity + low net displacement |

#### compare_trajectories

Compare two passages structurally. Returns a fitness score (lower = closer match).

```json
{
  "golden_text": "string (required)",
  "synthetic_text": "string (required)",
  "acceleration_threshold": "number (default: 0.3)"
}
```

Fitness components: DTW on acceleration profiles, Pearson correlation, spike position/count matching.

| Fitness | Meaning |
|---------|---------|
| < 0.3 | Excellent structural match |
| 0.3–0.5 | Good match, some rhythm deviation |
| 0.5–0.7 | Moderate — structure present but weak |
| > 0.7 | Poor match |

#### analyze_axis_alignment

Project a passage onto a caller-defined semantic axis and z-score the projection against a background corpus. Where `analyze_trajectory` measures *reflexive* geometry (how a passage moves relative to itself), this measures *referential* geometry (how strongly it marches along a direction **you** specify, e.g. escalation, formality, certainty).

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

A background null is **required** — z-scores are meaningless without it. Build one once per backend with `scripts/build_axis_null.py` (your corpus stays local; the cache is not committed). Returns:

| Field | Description |
|-------|-------------|
| `position_zscores` | Per-sentence position on the axis, in sigma units relative to the null |
| `axis_drift` | Net signed march along the axis (`z_last − z_first`) |
| `axis_straightness` | Discipline of the march along the axis (1.0 = straight line, 0.0 = oscillation) |
| `mean_zscore` | Mean axis position across the passage |
| `pole_separation` | `‖e₊ − e₋‖`; an "axis underdetermined" error fires if anchors embed too close |
| `null_count` | Number of background embeddings the null was built from |

The math for each computation is documented in [`docs/axis-alignment.md`](docs/axis-alignment.md); the design rationale is in [ADR-001](docs/ADRs/proposed/ADR-001-referential-axis-alignment.md).

#### model_status

Report current backend state: type, model name, dimensions, cache size. No parameters.

#### model_load

Load a specific embedding backend.

```json
{
  "backend": "nv_embed | lmstudio | sentence_transformers",
  "options": "object (optional backend-specific config)"
}
```

#### model_unload

Unload current model and clear embedding cache. Frees GPU memory. No parameters.

### Error Format

All tools return errors as:

```json
{
  "error": "Description of what went wrong"
}
```

## Trajectory Analysis

Treats text as a particle moving through embedding space. Each sentence is a point; the path between them encodes rhetorical structure.

### Metrics

| Metric | Definition | What it measures |
|--------|-----------|-----------------|
| Velocity | `‖e[i+1] - e[i]‖` | Pacing — magnitude of semantic shift between sentences |
| Acceleration | `\|v[i+1] - v[i]\|` | Rhythm — rate of pacing change |
| Curvature | Angular deflection between consecutive displacement vectors | Direction change in full embedding space |

### Spike Detection

An acceleration spike fires when `a[i] >= threshold` (default 0.3). Each spike records:

- **Index**: Position in sentence sequence
- **Magnitude**: Raw acceleration value
- **Isolation score**: How much the spike stands out from neighbors
- **Position ratio**: Where in the passage it occurs (0.0 = start, 1.0 = end)

### Composite Scores

**Deadpan score** (0–1): Isolated acceleration spikes against a stable background. Few spikes, high isolation, low background noise, strong contrast.

```
deadpan = 0.25 × spikiness + 0.35 × mean_isolation + 0.20 × background_stability + 0.20 × contrast
```

**Heller score** (0–1): Circular structure with deceleration. High pairwise similarity, low net displacement, negative velocity trend.

```
heller = 0.35 × circularity + 0.40 × tautology_density + 0.25 × deceleration
```

### Comparison / Fitness

`compare_trajectories` scores how well one passage matches another's structure (lower = better):

- DTW on acceleration profiles
- Pearson correlation of interpolated acceleration
- Spike position and count matching
- Weighted toward spike isolation quality (30%)

### Context Window Smoothing

The Gradio UI supports a sliding context window that averages N consecutive sentence embeddings before computing metrics. This smooths out filler (verbal tics, short interjections) without re-embedding.

`smoothed[i] = mean(e[i], e[i+1], ..., e[i+w-1])`

Window size 1 = no smoothing (default).

### Known Limitation

Velocity collapses 4096D displacement to a scalar (L2 norm), discarding direction. Acceleration compounds this. The PCA and heatmap visualizations compensate by operating on the full embedding matrix.

## Axis Alignment

Trajectory analysis is *reflexive* — it measures how a passage moves relative to itself. In high-dimensional space (NV-Embed-v2 is 4096D) that runs into a wall: independently varying vectors are nearly orthogonal by default, so inter-step angles saturate and curvature carries little signal.

Axis alignment is *referential*. You define a semantic direction with anchor exemplars, and the passage is projected onto that fixed axis. Here high dimensionality flips from liability to asset: the background projection concentrates tightly around its mean, so a genuine sustained march along the axis stands out at high sigma.

The instrument returns three things from one projection:

- **Position trace** — where each sentence sits on the axis, z-scored against a background corpus.
- **Axis drift** — the net signed march from first sentence to last.
- **Axis-restricted straightness** — whether the march is a disciplined straight line or an oscillation.

Two cautions are built into the tool:

- **Anisotropy.** Embeddings cluster in a narrow cone, so raw dot products are biased. Significance is therefore always a z-score against an empirical null, never an absolute alignment. Omitting `anchor_negative` uses the null mean as the negative pole, which de-means the cone in the same step.
- **The null is the experiment.** The z-score means "relative to *this* background population." Choose it deliberately — a real-conversation corpus and a literary corpus produce different sigmas for the same passage.

Full mathematical detail per function is in [`docs/axis-alignment.md`](docs/axis-alignment.md); the design rationale and trade-offs are in [ADR-001](docs/ADRs/proposed/ADR-001-referential-axis-alignment.md).

### Building a background null

```bash
# one segment per line, or a directory of .txt files
python scripts/build_axis_null.py corpus.txt --out cache/null.npy
export AXIS_NULL_MANIFEST=cache/null.npy.json
```

The cache is keyed by model name; rebuild it when you switch backends. Your corpus and the generated cache stay local — neither is committed.

## Gradio UI

Two tabs:

- **Drift** — Pairwise cosine distance between texts
- **Trajectory** — Analyze single passages or compare two. Interactive Plotly visualizations: velocity/acceleration/curvature profiles, PCA 2D projection, cosine similarity heatmap. Adjustable acceleration threshold and context window smoothing.

```bash
python -m semantic_kinematics
# Opens at http://localhost:7860
```

## Project Structure

```
semantic_kinematics/
├── embeddings/        # NV-Embed-v2, LM Studio, SentenceTransformers adapters
├── mcp/
│   ├── server.py      # MCP entry point
│   ├── state_manager.py
│   └── commands/      # embeddings, classification, trajectory, axis_alignment, model
├── ui/
│   ├── app.py         # Gradio application
│   ├── state.py       # Session state
│   └── tabs/          # drift, trajectory
└── utils/             # Text cleaning, HTML extraction

scripts/build_axis_null.py   # Build a background null cache for axis alignment
docs/                        # ADRs and math references (axis-alignment.md)
tests/                       # pytest suite
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for NV-Embed-v2 backend)
- See `pyproject.toml` for full dependency list

## License

MIT
