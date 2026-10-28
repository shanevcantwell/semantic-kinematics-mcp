# SK Gate-4 canonical text/vector remediation evidence

## Preregistration — 2026-10-28T14:46:07Z

This hypothesis and prediction were recorded before the real canonical smoke, regenerated corpus report, or post-remediation test runs below. Results are appended without changing this section.

### Question

Does the Gate-4 remediation preserve the intended semantic-kinematics consumer behavior while moving TVI text/ID authority to the verified canonical query door and leaving model-keyed embeddings in checkpoint vector state?

### Fixed inputs

- Canonical query executable: `/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus`
- Canonical DB: `/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db`
- Trusted canonical root: `/home/node/worktrees/tvi-canonical-query-door/output/canonical`
- Canonical acquisition scope: `input_key=base`
- Specimen-B conversation: `e7c2fe94-7960-414d-8d59-92fdd2ee2303`, all acquisitions, speakers `user,assistant`, stable ascending `message_index`
- Embeddinggemma-era checkpoint: `/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/embed_checkpoint.jsonl`, SHA-256 `fb75d130e732ec6e7b3b919acd50b6600a2821920c8971ab01edc88cbaa35305`
- NV-Embed-v2 checkpoint: `/mnt/storage/thought-vault-integration-data/vectors/nv4096/corpus_4096.jsonl`, SHA-256 `e87ec2b2c139f53d64c6fb00d18b93da2ec2a7603f8c83546968d0de117fa281`
- Legacy parity reference: `/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/chunks.jsonl`

### H0

The remediation does not close Gate 4. At least one required property fails: the canonical smoke selects a different specimen-B population/order, one or more selected IDs cannot join to checkpoint vectors, the generated corpus report derives text totals from a legacy checkpoint/JSONL surface or omits canonical generation/query provenance, focused regressions fail, or the full suite fails.

### Prediction

H0 will be rejected. Specifically:

1. `smoke_jolt --specimen-b-only` will query canonical text/metadata through the fixed executable/DB/root, report the verified generation and query fingerprint, select 201 ordered turns, join all 201 IDs to 768-dimensional checkpoint vectors, and reproduce the established specimen-B trajectory verdict without contacting a live embedding endpoint.
2. `scripts/summarize_corpus.py` will regenerate `docs/generated/CORPUS_STATS.md` from canonical `base` IDs/totals plus the two model-keyed checkpoints; each store section will include canonical generation ID, database/logical digests, query verb/ordering/fingerprint, and explicit checkpoint metadata/source-binding status.
3. The real-generation parity suite will preserve exact ordered consumer parity, including the smoke-jolt specimen sequence.
4. The focused Gate-4 suite and the full repository suite will pass.

### Acceptance rule

PASS only if the real canonical smoke exits zero with every selected turn joined, the committed generated report names canonical generation/query provenance and honest checkpoint binding/metadata state, all real-generation ordered comparisons are exact, and both focused and full test commands exit zero. No fallback to TVI `chunks.jsonl` is allowed in the smoke or summarizer; the legacy file is permitted only as the explicit parity oracle in the opt-in parity test.

## Commands and results

### 1. Real canonical checkpoint smoke

Command:

```bash
env \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  THOUGHT_VAULT_VECTORS_DIR=/home/node/github/shanevcantwell/thought-vault-integration/output/vectors \
  python scripts/smoke_jolt.py --specimen-b-only
```

Result: **PASS**.

- canonical generation: `tvi-generation-v1:sha256:5e3cae0cf5d8ecd5a91608301fed6d7bee5c5cd717d0cf8c23c8b886a0e30be3`
- logical digest: `161e1a8d643b9228e40a1c7b1233fd98f6822994cac101e077bc134e65a9c380`
- scoped query: `iter_legacy_chunks` / `whole_corpus_v1`
- query fingerprint: `b872ea1b85fa8ce91882fdc7377e5ebf058c25514c0907ab26dda84d1d420c9f`
- selected and joined: `201/201` turns, `768` dimensions
- checkpoint SHA-256: `fb75d130e732ec6e7b3b919acd50b6600a2821920c8971ab01edc88cbaa35305`
- exact selected `(chunk_id,text)` sequence SHA-256: `9ccfe2e83268b858cb27a7298187c43ca606492bf9fda93f26ab06d0950f5f07`
- provenance disposition: accepted only by the pinned historical exception (checkpoint digest + canonical generation/scoped-query fingerprint + exact sequence parity); producing model identity remains non-authoritative
- max acceleration: `0.5980` at index `147`
- specimen-B verdict: `ISOLATED SPIKE present` (`peak_to_median=5.98`, `top_isolation=0.9519`, `deadpan=0.7463`, `n_spikes=13`)

No live embedding endpoint was contacted; text/metadata came from the canonical query and vectors came from the checkpoint. This result is specimen-B-only. The default cross-specimen/shared-space path now refuses this historical checkpoint because it lacks authoritative model metadata; a combined verdict requires matching `.meta.json` and `.source.json` sidecars.

### 2. Canonical corpus summary regeneration

Command:

```bash
env \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  SK_CHECKPOINT_EMBGEMMA_300M_768=/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/embed_checkpoint.jsonl \
  SK_CHECKPOINT_NVEMBED_V2_4096=/mnt/storage/thought-vault-integration-data/vectors/nv4096/corpus_4096.jsonl \
  python scripts/summarize_corpus.py
```

Result: **PASS**; `docs/generated/CORPUS_STATS.md` was regenerated.

Observed canonical query provenance for both stores:

- generation: `tvi-generation-v1:sha256:5e3cae0cf5d8ecd5a91608301fed6d7bee5c5cd717d0cf8c23c8b886a0e30be3`
- database SHA-256: `1be54c9c9af38defa75525220bbc8177440a5a508aef368fc13dfdcc22818f17`
- logical digest SHA-256: `161e1a8d643b9228e40a1c7b1233fd98f6822994cac101e077bc134e65a9c380`
- query: `iter_legacy_chunks` / `whole_corpus_v1`
- query fingerprint: `43ae975f2213c9473603ce69589fee4d3d97d4e80e31bd3c3ed463dc4e86f78e`
- canonical total: `87,004`

Observed checkpoint vector state:

| Store | Covered IDs | Failed | Uncovered | Dim | Source binding | Metadata | Checkpoint SHA-256 |
|-------|------------:|-------:|----------:|----:|----------------|----------|-------------------|
| embeddinggemma-era | 87,004 | 0 | 0 | 768 | legacy-unbound; no `.source.json` | dimension inferred; model identity not authoritative | `fb75d130e732ec6e7b3b919acd50b6600a2821920c8971ab01edc88cbaa35305` |
| NV-Embed-v2 | 86,748 | 256 | 256 | 4096 | legacy-unbound; no `.source.json` | authoritative `.meta.json` | `e87ec2b2c139f53d64c6fb00d18b93da2ec2a7603f8c83546968d0de117fa281` |

The report makes no claim that either historical checkpoint was produced by the current canonical generation. It labels the metric as checkpoint ID coverage against the queried generation, reserves completion wording for a validated source binding, and durably identifies each checkpoint by content hash.

Two pre-fix diagnostic attempts correctly exposed report-path defects before the successful regeneration:

```text
ValueError: checkpoint '/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/embed_checkpoint.jsonl' is populated but has no source provenance sidecar '/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/embed_checkpoint.jsonl.source.json'; refusing unsafe legacy adoption
```

and, after separating report observation from resume adoption:

```text
OSError: [Errno 30] Read-only file system: '/mnt/storage/thought-vault-integration-data/vectors/nv4096/corpus_4096.jsonl.source.json.lock'
```

The repaired summarizer now observes legacy-unbound checkpoints without minting sidecars or lock files, validates a source sidecar when one exists, and acquires metadata, legacy dimension inference, digest, and records from one descriptor-bound snapshot. It compares `fstat` before/after plus final path device/inode/size/mtime/ctime identity, with regressions for atomic replacement and same-size rewrites with restored mtime.

### 3. Real-generation ordered parity

Command:

```bash
env \
  SK_RUN_REAL_TVI_PARITY=1 \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  python -m pytest -s -m real_generation tests/test_real_tvi_generation_parity.py
```

Result: **4 passed in 42.89s**.

The added smoke-jolt comparison observed `201` turns and sequence SHA-256 `9ccfe2e83268b858cb27a7298187c43ca606492bf9fda93f26ab06d0950f5f07`, identical between canonical and legacy selection/order. Existing null-builder and embedding-input hashes also remained unchanged.

### 4. Focused Gate-4 suite

Command:

```bash
env \
  SK_RUN_REAL_TVI_PARITY=1 \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  python -m pytest -q \
    tests/test_smoke_jolt_canonical_input.py \
    tests/test_summarize_corpus.py \
    tests/test_no_hardcoded_paths.py \
    tests/test_real_tvi_generation_parity.py
```

Final result after review regressions: **30 passed in 44.08s**.

The first focused invocation exposed that test-isolation defect exactly as:

```text
AssertionError: scripts.build_displacement_null.DEFAULT_CORPUS_DB = /home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db
```

The test now removes the override before asserting the built-in `/srv/` defaults.

### 5. Full suite

Command:

```bash
env \
  SK_RUN_REAL_TVI_PARITY=1 \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  python -m pytest
```

Final result after review remediation: **509 passed in 48.40s**.

## Analysis

Reject H0 for this remediation and promoted generation. The real specimen-B smoke exercised the intended split—canonical text/metadata plus checkpoint vectors—and joined the complete established specimen population under an exact, documented legacy artifact exception. It does not launder that exception into a shared embedding-space claim: the default combined path refuses until authoritative model/source sidecars exist. The generated report carries canonical generation/query provenance, stable-snapshot checkpoint hashes, and honest unbound ID-coverage labels. Ordered real-generation parity, replacement-race regressions, focused tests, and the full suite all passed.
