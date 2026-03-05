<img alt="semantic-kinematics-mcp" src="img/semantic-kinematics-banner.png" />
Embedding space analysis toolkit. Measures semantic drift between texts, traces trajectory dynamics through prose, and exposes everything as MCP tools for agentic integration.

## Quick Start

```bash
# Install
pip install -e .

# GPU support (NV-Embed-v2, ~14GB VRAM)
pip install -e ".[gpu]"

# Launch Gradio UI
python -m semantic_kinematics

# Or start MCP server
semantic-kinematics-mcp
```

### Docker

```bash
docker build -t semantic-kinematics-mcp .
docker run -i --rm semantic-kinematics-mcp
```

Or with docker-compose for host networking and data mounts:

```bash
docker-compose up
```

## Embedding Backends

Three interchangeable backends, selected via `EMBEDDING_BACKEND` environment variable:

| Backend | Model | Dimensions | Notes |
|---------|-------|------------|-------|
| `nv_embed` | NV-Embed-v2 | 4096 | GPU, fp16, highest quality |
| `lmstudio` | Any GGUF via OpenAI API | Varies | Local LM Studio server |
| `sentence_transformers` | Any HuggingFace model | Varies | General purpose |

Configure in `.env`:

```
EMBEDDING_BACKEND=nv_embed
```

## MCP Tools

8 tools over JSON-RPC (stdio).

| Tool | Description |
|------|-------------|
| `embed_text` | Get embedding vector for text |
| `calculate_drift` | Cosine distance between two texts |
| `classify_document` | Similarity-based document classification |
| `analyze_trajectory` | Velocity, acceleration, curvature metrics for a passage |
| `compare_trajectories` | Fitness score: compare two passages structurally |
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

## Gradio UI

Two tabs:

- **Drift** — Pairwise cosine distance between texts
- **Trajectory** — Analyze single passages or compare two. Interactive Plotly visualizations: velocity/acceleration/curvature profiles, PCA 2D projection, cosine similarity heatmap. Adjustable acceleration threshold and context window smoothing.

```bash
python -m semantic_kinematics
# Opens at http://localhost:7861
```

## Project Structure

```
semantic_kinematics/
├── embeddings/        # NV-Embed-v2, LM Studio, SentenceTransformers adapters
├── mcp/
│   ├── server.py      # MCP entry point
│   ├── state_manager.py
│   └── commands/      # embeddings, classification, trajectory, model
├── ui/
│   ├── app.py         # Gradio application
│   ├── state.py       # Session state
│   └── tabs/          # drift, trajectory
└── utils/             # Text cleaning, HTML extraction
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for NV-Embed-v2 backend)
- See `pyproject.toml` for full dependency list

## License

MIT
