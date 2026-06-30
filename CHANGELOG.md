# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this project follows
[Semantic Versioning](https://semver.org/) (pre-1.0 `0.x`: minor = features,
patch = fixes; `-alpha` denotes a pre-release not yet validated end-to-end).

## [0.3.0-alpha] — 2026-06-30

First tagged pre-release. Headline: the **nv_embed bulk-embedding pipeline** went
from non-functional to a working, resumable, self-describing data plane. This is
an *alpha* — the embedding substrate produces data, but the downstream geometry
(axis-alignment null built + analysis validated on real data) is not yet
demonstrated end-to-end; that milestone gates a release candidate.

### Added
- `--backend nv_embed` now works end-to-end through `scripts/embed_corpus.py`;
  it previously raised `TypeError` on adapter kwargs and failed every item for
  lack of `count_tokens`. (#40)
- `NVEmbedAdapter.count_tokens` — tokenizer-only load, no model weights. (#40)
- `scripts/embed_status.py` — truthful `done failed pending total` progress
  signal (the wrapper trusts this, not the process exit code). (#40)
- `scripts/embed_full_corpus.sh` — resumable full-corpus runner: auto-restart on
  crash, success-count completion, nv_embed-appropriate defaults. (#41)
- Self-describing checkpoints: a sidecar `<checkpoint>.meta.json` records
  `model_name` + `dimensions`; resuming a checkpoint built by a different model
  fails loud instead of silently merging incompatible vectors. (#16)
- The `nv_embed` backend now honors `NV_EMBED_MODEL_PATH` (previously only
  `sentence_transformers` did). (#43)

### Changed
- `BulkEmbedder` streams prep + embed in windows (`prep_window`, default 256)
  rather than prepping the whole corpus upfront — prep is now resumable, so the
  *entire* run (not just embedding) reconstructs from the checkpoint. (#42)
- nv_embed bulk runs hold the model resident (`unload_after_use=False`) and
  default to 8000-token budgets, vs the embeddinggemma-sized 1500/3000 that
  over-split nv_embed's 32k context. (#40, #41)
- `README.md` and `docs/ARCHITECTURE.md` synced to the current embeddings/bulk
  reality (new scripts, nv_embed guidance, windowed prep, env-var scope). (#43)

### Fixed
- `calculate_drift` repointed to the canonical `adapter.cosine_distance` (was
  importing a module that never existed). (#35)
- Hardcoded `/home/shane` paths in the sentence-transformers adapter and the
  null-building scripts are now environment-driven. (#34)
