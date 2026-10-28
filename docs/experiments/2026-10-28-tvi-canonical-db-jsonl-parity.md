# TVI canonical DB ↔ legacy JSONL consumer parity

## Preregistration — 2026-10-28T06:15:41Z

This section was written before executing the real-generation parity test. Results are appended below rather than changing the hypothesis after observation.

### Question

Does the promoted TVI canonical database reproduce the legacy `output/vectors/chunks.jsonl` consumer populations exactly when semantic-kinematics reads text and chunk IDs through the installed canonical query door?

### Fixed inputs

- Query executable: `/home/node/worktrees/tvi-canonical-query-door/.venv/bin/tvi-canonical-corpus`
- Promoted DB: `/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db`
- Trusted canonical root: `/home/node/worktrees/tvi-canonical-query-door/output/canonical`
- Canonical acquisition scope: `input_key=base`
- Legacy JSONL: `/home/node/github/shanevcantwell/thought-vault-integration/output/vectors/chunks.jsonl`
- Null-builder usability rule: stripped text length ≥ 40 characters
- Reservoir sample: `k=3000`, seed `1729`
- Embedding-input rule: skip whitespace-only text; preserve source text bytes otherwise; pair each item with its `chunk_id`

### H0

The canonical DB query and legacy JSONL do **not** yield identical consumer inputs. At least one of these differs: usable-text order, usable count, fixed-seed reservoir sample, or ordered `(chunk_id, text)` embedding inputs.

### Prediction

H0 will be rejected: `input_key=base` queried with `whole_corpus_v1` ordering will yield the same ordered legacy population. Therefore:

1. both measured-null builders will see byte-identical usable-text sequences;
2. both will produce identical `k=3000`, seed-`1729` reservoir samples and usable counts; and
3. `embed_corpus.load_corpus_source` will produce identical ordered `(chunk_id, text)` inputs in canonical-DB and generic-JSONL modes.

### Acceptance rule

PASS only if every ordered comparison is exact. No set-only comparison, sorting, tolerance, or fallback is allowed. The test must use the installed CLI and verified-open DB path above. It must stop before adapter construction or embedding generation; existing vector checkpoints are not refreshed or rewritten.

## Results — 2026-10-28

Command (from the sk-mcp worktree):

```bash
env \
  SK_RUN_REAL_TVI_PARITY=1 \
  TVI_CANONICAL_CORPUS_EXECUTABLE=/tmp/tvi-main-cli-smoke-venv/bin/tvi-canonical-corpus \
  THOUGHT_VAULT_CANONICAL_DB=/home/node/worktrees/tvi-canonical-query-door/output/canonical/corpus.db \
  THOUGHT_VAULT_CANONICAL_ROOT=/home/node/worktrees/tvi-canonical-query-door/output/canonical \
  python -m pytest -s -m real_generation tests/test_real_tvi_generation_parity.py
```

Result after the Gate-4 smoke-jolt parity case was added: **4 passed in 42.89s**.

Observed canonical generation:

- `generation_id`: `tvi-generation-v1:sha256:5e3cae0cf5d8ecd5a91608301fed6d7bee5c5cd717d0cf8c23c8b886a0e30be3`
- canonical rows: `87,004`
- ordered embeddable `(chunk_id, text)` inputs: `87,004`
- embedding-input sequence SHA-256 (length-framed UTF-8 fields): `9c595debc21debdbfacb747e07bf152934a85145c62f91f9d3d509b146225c81`

Both null builders independently produced the same observations:

- usable texts (stripped length ≥ 40): `62,372`
- usable-text sequence SHA-256 (length-framed UTF-8): `b6a1cea9e5fac4defe0f0637482232327271d54928a4de53e630c7be5d82a006`
- reservoir sample: `k=3000`, seed `1729`
- sample sequence SHA-256 (length-framed UTF-8): `ffc09b2749796fdfb78b9e15aa8b3965e5686288ecefe5f033696bafb3e31258`

The smoke-jolt specimen-B population also matched exactly:

- selected user/assistant turns: `201`
- ordered `(chunk_id, text)` sequence SHA-256: `9ccfe2e83268b858cb27a7298187c43ca606492bf9fda93f26ab06d0950f5f07`
- scoped query fingerprint: `b872ea1b85fa8ce91882fdc7377e5ebf058c25514c0907ab26dda84d1d420c9f`

Every comparison was ordered and exact:

1. canonical and legacy usable-text sequences matched item-for-item;
2. usable counts and fixed-seed samples matched exactly for both `build_displacement_null` and `build_conditioned_null`;
3. canonical and legacy `smoke_jolt` specimen-B selection/order matched item-for-item; and
4. canonical and legacy `embed_corpus` inputs matched item-for-item as ordered `(chunk_id, text)` pairs.

The embedding-input test constructed no adapter and opened no vector checkpoint. No embedding or model-sidecar file was created, updated, or refreshed.

## Analysis

Reject H0 for this promoted generation and query scope. The prediction held across all preregistered surfaces: the verified canonical DB query door conserves the legacy text/ID population and ordering required by the null samplers and bulk embedder.

This result is deliberately narrower than vector equality. It certifies the **consumer-input axis** only: canonical text/IDs, usability filtering, deterministic sampling, and embedding-call inputs. Existing embeddings remain model-keyed derived state and were not regenerated. Model identity/dimension continue to be enforced by `<checkpoint>.meta.json`; canonical generation/query identity is enforced separately by `<checkpoint>.source.json`. Generic JSONL remains valid for non-TVI corpora, including the Nemotron contrast legs, rather than becoming a fallback for failed canonical TVI acquisition.
