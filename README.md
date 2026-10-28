<img width="1920" height="1072" alt="image" src="https://github.com/user-attachments/assets/eeb237a7-2fc6-414d-965b-26a116e8a739" />

# semantic-kinematics-mcp

A surface **embedding-vector analysis** toolkit. It takes the vectors an embedding model hands back and holds them up to different lights — semantic drift, trajectory dynamics, projection onto caller-defined axes, displacement-magnitude jolts — to see what structure falls out, cheaply. Everything is exposed as MCP tools for agentic integration.

> **Scope, honestly:** this works on the embed model's *output vectors only* — no transformer hidden-state access. It is the cheap, surface-level instrument; the question it answers is *"what can you measure with just the vectors?"* Reaching into hidden states is a later, separate bet.

## What it offers

| Light | What it measures |
|-------|------------------|
| **Drift** | Cosine distance between two texts — the simplest, most validated signal. |
| **Trajectory** | Text as a particle through embedding space: velocity, acceleration, curvature, and composite rhythm scores. *Reflexive* geometry — how a passage moves relative to itself. |
| **Axis alignment** | Project a passage onto a semantic direction *you* define (escalation, formality, certainty…), z-scored against an empirical null. *Referential* geometry — how strongly it marches along your axis. |
| **Bearing / jolt** *(library-level, emerging — not yet a contracted MCP tool)* | Axis-free displacement-magnitude jolt detection scored against a measured-displacement null. |
| **Classification** | Similarity-based document classification against caller-supplied exemplars. |

Plus a **bulk embedding engine** (`BulkEmbedder`) for turning a corpus into vectors at scale, and three interchangeable **embedding backends** behind one adapter contract.

Method math, full per-tool request/response schemas, the layering invariant, and the data pipeline live in **[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)**.

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
# or, for host networking + data mounts:
docker-compose up
```

## Embedding Backends

Three interchangeable backends behind one `EmbeddingAdapter` (this is the "switching layer" — multiple providers' schemas normalized to one contract). Backend selection has two paths: the **MCP server** reads the `EMBEDDING_BACKEND` environment variable, while the **`embed_corpus.py` bulk CLI** selects via its `--backend` flag (it does not consult `EMBEDDING_BACKEND`):

| Backend | Model | Dimensions | Notes |
|---------|-------|------------|-------|
| `nv_embed` | NV-Embed-v2 | 4096 | GPU, fp16 (~14GB VRAM), highest quality. Custom `BidirectionalMistralModel`; resists GGUF/quantization — fp16 in-process only. |
| `lmstudio` | Any GGUF via OpenAI API | Varies | Local LM Studio / llama-server endpoint. |
| `sentence_transformers` | Any HuggingFace model | Varies | General purpose, CPU-friendly with a small model. |

```
# .env
EMBEDDING_BACKEND=nv_embed
```

Path resolution is environment-driven (no hardcoded home directories — issue #34):

| Variable | Used by | Default |
|----------|---------|---------|
| `NV_EMBED_MODEL_PATH` | both in-process backends (`nv_embed` + `sentence_transformers`) | `nvidia/NV-Embed-v2` (HuggingFace hub id; set to a local checkout to skip download) |
| `THOUGHT_VAULT_CANONICAL_DB` | measured-null builders, `smoke_jolt`, corpus summary | `/srv/dev/shanevcantwell/thought-vault-integration/output/canonical/corpus.db` |
| `THOUGHT_VAULT_CANONICAL_ROOT` | canonical TVI consumers (optional trusted root for direct-generation paths) | unset |
| `TVI_CANONICAL_CORPUS_EXECUTABLE` | canonical TVI consumers | `tvi-canonical-corpus` |
| `THOUGHT_VAULT_VECTORS_DIR` | `smoke_jolt` checkpoint-vector boundary only | `/srv/dev/shanevcantwell/thought-vault-integration/output/vectors` |

## MCP Tools

9 tools over JSON-RPC (stdio). Full request/response schemas and return-field tables are in **[ARCHITECTURE.md → Tool reference](docs/ARCHITECTURE.md#tool-reference)**.

| Tool | Description |
|------|-------------|
| `embed_text` | Embedding vector for text |
| `calculate_drift` | Cosine distance between two texts |
| `classify_document` | Similarity-based document classification |
| `analyze_trajectory` | Velocity / acceleration / curvature metrics for a passage |
| `compare_trajectories` | Fitness score comparing two passages structurally |
| `analyze_axis_alignment` | Project a passage onto a caller-defined axis, z-scored against a background null |
| `model_status` | Backend state (type, model, dimensions, cache) |
| `model_load` | Load a backend *(slated for removal under ADR-003)* |
| `model_unload` | Unload model, free memory *(slated for removal under ADR-003)* |

### Configure in Claude Code

```json
{
  "mcpServers": {
    "semantic-kinematics": {
      "command": "semantic-kinematics-mcp",
      "env": { "EMBEDDING_BACKEND": "nv_embed" }
    }
  }
}
```

## Bulk Embedding a Corpus

`BulkEmbedder` (`semantic_kinematics/embeddings/bulk.py`) turns an ordered text/ID corpus into vectors at scale. For thought-vault data, text and `chunk_id` now come through TVI's verified canonical DB query door; generic/non-TVI JSONL remains an explicit supported input mode. The embedder wraps any backend and adds **windowed crash-resume** (it streams prep + embed in windows of `prep_window` items, default 256 — checkpointing each embedded item to JSONL, so an interrupted run re-preps only the not-yet-checkpointed remainder and the *whole* run, prep included, reconstructs from the checkpoint rather than restarting from scratch), **token-aware batching** (packs requests under a token budget), **sub-chunking with vector averaging** (splits over-long items, averages the piece vectors), and **backoff retries**.

Canonical TVI input (repeat `--input-key` to combine explicit acquisition legs):

```bash
python scripts/embed_corpus.py \
    --corpus-db /path/to/output/canonical/corpus.db \
    --canonical-root /path/to/output/canonical \
    --input-key base \
    --tvi-executable /path/to/bin/tvi-canonical-corpus \
    --checkpoint out.jsonl \
    --backend lmstudio \
    --base-url http://localhost:8082/v1 \
    --model embeddinggemma-300M-F32
```

Generic JSONL remains available for new contrast corpora and other non-TVI data:

```bash
python scripts/embed_corpus.py corpus.jsonl \
    --checkpoint out.jsonl \
    --backend lmstudio \
    --base-url http://localhost:8082/v1 \
    --model embeddinggemma-300M-F32 \
    --text-field text \
    --id-field chunk_id \
    --max-tokens-per-request 3000 \
    --max-tokens-per-chunk 1500
```

For the in-process **`nv_embed`** backend (NV-Embed-v2 @ 4096-d), two things differ. The model is held **resident** for the whole run — `embed_corpus.py` sets `unload_after_use=False` automatically for `nv_embed`, since the per-call unload default would reload ~15GB of weights per request-group and make a corpus-scale run prohibitively slow. And bulk runs use **larger token budgets** — nv_embed's context is 32768, so the embeddinggemma-sized defaults (1500/3000) over-split the long tail; pass `--max-tokens-per-request 8000 --max-tokens-per-chunk 8000` to keep ~99.7% of the corpus a single piece:

```bash
python scripts/embed_corpus.py corpus.jsonl \
    --checkpoint out.jsonl \
    --backend nv_embed \
    --max-tokens-per-request 8000 \
    --max-tokens-per-chunk 8000
```

For a full thought-vault corpus, **`scripts/embed_full_corpus.sh`** is the canonical runner: it defaults to the canonical DB (`CORPUS_DB`, `INPUT_KEYS=base`) and wraps the above with auto-restart on crash and a success-count-based completion signal (via `scripts/embed_status.py`) so a run that crashes mid-way resumes cleanly and never false-completes on `_failed` items. Set `CORPUS=/path/to/corpus.jsonl` instead for an explicit generic JSONL run; `CORPUS` and `CORPUS_DB` are mutually exclusive. `scripts/embed_status.py` prints `done failed pending total` for either source mode and the same checkpoint.

- **Canonical TVI input:** `--corpus-db` plus one or more `--input-key` values. The external `tvi-canonical-corpus` CLI performs verified-open and streams rows in canonical query order; a query/provenance failure is fatal and never falls back to legacy JSONL.
- **Generic JSONL input:** one object per line; `--text-field` / `--id-field` name the text and id keys (blank-text lines are skipped; missing ids default to `line-N`). This mode remains the contract for Nemotron and other newly prepared contrast corpora.
- **Vector boundary:** the `--checkpoint` JSONL remains model-keyed derived state: one `{chunk_id, embedding}` record per item. `<checkpoint>.meta.json` records the producing model (`model_name` + `dimensions`), and a model mismatch fails loud (#16). Corpus generation/query identity is deliberately separate in atomic `<checkpoint>.source.json`; resume refuses a changed canonical generation/query (or changed generic file hash) without putting source metadata into each vector record.
- **Where it sits:** `BulkEmbedder` is a *data-plane* job on the shared adapter substrate — it never crosses the MCP contract boundary. Its output vectors feed downstream analysis, e.g. building the axis-alignment background null. The sibling **thought-vault-integration** repo owns ingestion and promotion of text/IDs into its canonical DB; sk-mcp reads them through the installed query CLI. See [ARCHITECTURE.md → Data pipeline](docs/ARCHITECTURE.md#data-pipeline-bulk-embedding).

`scripts/smoke_jolt.py` is **canonical text plus checkpoint vectors**: specimen-B conversation IDs, speakers, order metadata, and text come from an explicit verified canonical query (`input_key=base`, target conversation across acquisitions, user/assistant only), while vectors come from a provenance-gated checkpoint. Normal operation requires matching `.source.json` plus `.meta.json` sidecars. The historical sidecar-less specimen-B checkpoint is accepted only by a pinned checkpoint SHA-256, canonical generation/scoped-query fingerprint, and exact text/ID sequence parity; its producing model is explicitly non-authoritative, so it is available only through `python scripts/smoke_jolt.py --specimen-b-only` and cannot support the default cross-specimen/shared-space verdict. The default two-specimen run requires both authoritative checkpoint model identity matching specimen A and the live embeddinggemma endpoint. `scripts/summarize_corpus.py` likewise obtains TVI totals/IDs and generation/query provenance canonically while scanning checkpoint snapshots only for vector validity and source-ID coverage. Both fail closed on verified-open/query errors and never treat checkpoint JSONL as text authority.

Host-local release evidence for the DB cutover is opt-in because it depends on a promoted real generation:

```bash
SK_RUN_REAL_TVI_PARITY=1 python -m pytest -s -m real_generation \
    tests/test_real_tvi_generation_parity.py
```

That test compares the canonical DB against the legacy `output/vectors/chunks.jsonl` byte-for-byte in consumer order: usable null-builder text, fixed-seed reservoir samples, and ordered `(chunk_id, text)` embedding inputs. It constructs no adapter and refreshes no embeddings. The preregistration and observed results live in [`docs/experiments/2026-10-28-tvi-canonical-db-jsonl-parity.md`](docs/experiments/2026-10-28-tvi-canonical-db-jsonl-parity.md).

## Gradio UI

Two tabs — **Drift** (pairwise cosine distance) and **Trajectory** (analyze one passage or compare two; interactive velocity/acceleration/curvature profiles, PCA projection, cosine-similarity heatmap, adjustable threshold + context-window smoothing).

```bash
python -m semantic_kinematics   # opens at http://localhost:7860
```

## Project Structure

```
semantic_kinematics/
├── embeddings/        # EmbeddingAdapter + nv_embed / lmstudio / sentence_transformers backends
│                      # + BulkEmbedder (bulk.py): resumable, token-aware corpus embedding
│                      # + bearing/ : axis-free jolt + conditioned-atom (library-level, research)
├── mcp/
│   ├── server.py      # MCP entry point (sole contract door)
│   ├── state_manager.py
│   └── commands/      # embeddings, classification, trajectory, axis_alignment, model
├── ui/                # Gradio app + drift/trajectory tabs
└── utils/             # text cleaning, HTML extraction

scripts/build_axis_null.py   # build a background null cache for axis alignment
scripts/embed_corpus.py      # bulk-embed a corpus (resumable, token-aware)
scripts/embed_status.py      # report `done failed pending total` for a (corpus, checkpoint) pair
scripts/embed_full_corpus.sh # resumable full-corpus runner: auto-restart, success-count completion
docs/                        # ARCHITECTURE.md, axis-alignment.md, ADRs, HANDOFF.md
tests/                       # pytest suite
```

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** — layering invariant, tool reference, analysis methods (the math), data pipeline, conformance gaps.
- **[axis-alignment.md](docs/axis-alignment.md)** — full math for the referential axis projection.
- **[ADRs](docs/ADRs/proposed/)** — design decisions (ADR-001 axis alignment, ADR-002 unified adapter, ADR-003 stateless MCP).
- **[HANDOFF.md](docs/HANDOFF.md)** — cross-repo resume index.

## Requirements

- Python 3.10+
- PyTorch 2.0+ (for the NV-Embed-v2 backend)
- See `pyproject.toml` for the full dependency list.

## License

MIT
