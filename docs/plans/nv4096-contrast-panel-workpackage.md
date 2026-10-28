# Work-package: nv-embed-v2 contrast panel (Nemotron ↔ thought-vault)

**Audience:** a **non-frontier orchestrator model**. Written to be executed with minimal
open-ended judgment — reuse existing scripts, gate every phase on a deterministic check,
STOP-and-report on any failed gate rather than proceeding with a plausible substitute.

**Goal:** produce one or more Nemotron-derived embedded datasets through the *byte-identical*
nv-embed-v2 path that produced thought-vault's `nv4096`, so the two can be contrasted to
separate the **instrument cone** (nv-embed's intrinsic anisotropy) from **operator voice**.
The invariant across corpora is the instrument; what differs is voice.

---

## The non-negotiable spine — the shared-machine invariant

Every corpus in the panel MUST be embedded through the identical path, or the contrast
measures *two instruments* instead of voice. Identical means all of:

| Facet | Required value | Enforced by |
|---|---|---|
| Model id | `nvidia/NV-Embed-v2` (locally cached; no download) | `.meta.json` sidecar, written automatically |
| Dim | 4096 | sidecar |
| Normalization | `F.normalize(p=2, dim=1)` — unit sphere | `nv_embed_adapter.py:278,309` (do not change) |
| Pooling | NV-Embed's own `model.encode()` (latent-attention) | adapter (do not reimplement) |
| Chunk unit | **one chunk per message** (matches thought-vault) | canonical TVI rows for Phase 0; your Phase-2 chunker for new corpora |
| Text/ID authority | promoted TVI canonical DB, queried through `tvi-canonical-corpus` (`input_key=base`) | verified-open query provenance |
| Vector boundary | `{"chunk_id": str, "embedding": [4096 floats]}` + model `.meta.json`; no corpus text copied into checkpoint rows | `BulkEmbedder`, automatic |
| Hardware | **the RTX-8000 host** (same box that made nv4096) | you run there |

If any facet cannot be held identical, **STOP and report** which one and why. Do not
substitute a different embedder, dtype, pooling, or chunk unit to "make it work." A mismatch
silently invalidates the entire result.

## Reuse, do not reinvent

You will **not** write embedding code. The exact driver that produced nv4096 is reusable:

```bash
# from /srv/dev/shanevcantwell/semantic-kinematics-mcp
CORPUS=/path/to/new_texts.jsonl \
CKPT=/path/to/out/new_corpus_4096.jsonl \
LOG=/path/to/out/embed.log \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash scripts/embed_full_corpus.sh
```

- **New contrast-corpus input contract:** `CORPUS` is generic JSONL; each line has a `text`
  field and a `chunk_id` field. This mode remains supported because Nemotron and other external
  corpora are not TVI acquisitions.
- **Existing thought-vault text/ID contract:** never recover text or IDs from a vector file or a
  stale JSONL export. Query the promoted canonical DB through the installed
  `tvi-canonical-corpus` CLI. The concrete Phase-0 command is below.
- **Output contract (automatic):** `{chunk_id, embedding[4096]}` per line + a
  `<CKPT>.meta.json` sidecar `{"model_name":"NVEmbed:nvidia/NV-Embed-v2","dimensions":4096}`.
  The checkpoint and model sidecar remain model-keyed derived vector state. A separate atomic
  `<CKPT>.source.json` records source generation/query identity; do not copy text into vector rows
  or overload the model sidecar with corpus identity. Resume fails loud on either model/dim or
  source-generation/query mismatch — do not delete or bypass either sidecar.
- Your whole job is: **fetch → select exemplars → chunk to the generic JSONL contract → run the
  driver → verify → compact.** Thought-vault is used only as the canonical Phase-0 reference leg.

---

## Phases (each ends in a deterministic acceptance gate)

### Phase 0 — Prove the path FIRST (the invariant gate)
Before spending any compute on new data, prove your embedding path reproduces nv4096's.

1. From the existing nv4096 checkpoint, pick **20 chunk_ids** whose stored vector is real
   (not `_failed`, not zero-norm). Treat this file only as the model-keyed vector/reference leg.
2. Recover the matching source `text` and authoritative `chunk_id` from the promoted TVI DB —
   not from `output/vectors/chunks.jsonl` and not from the vector checkpoint:

   ```bash
   /path/to/tvi-canonical-corpus query chunks \
       --db /path/to/output/canonical/corpus.db \
       --canonical-root /path/to/output/canonical \
       --format jsonl --all-pages --input-key base
   ```

   Verify the metadata record declares `verified_open=true`, retain its generation/query
   provenance in the Phase-5 note, and join the 20 selected vector IDs to canonical rows by
   `chunk_id`. Missing or duplicate matches are a hard STOP.
3. Write only those 20 canonical `(chunk_id, text)` pairs to a scratch generic JSONL file, then
   embed them through the driver above into a fresh scratch `CKPT`. This bounded JSONL bridge is
   allowed because the canonical query already established the text/ID authority; it does not
   make legacy JSONL authoritative again.
4. For each, compute `cosine(new_vector, stored_vector)`.

**GATE:** all 20 cosines ≥ **0.9999**. (Same path → same vectors; nv-embed fp16 is
effectively deterministic.) If any diverge, your path is NOT identical — STOP and report the
suspect facet (model path? dtype? device? a code edit?). **Do not proceed to Phase 1 until
this passes.** This is the cheapest possible falsification of "am I on the same instrument."

### Phase 1 — Fetch the datasets
Target sets (HuggingFace, `nvidia/…`): `Nemotron-RL-Identity-Following-v1` (21.7k),
`Nemotron-RL-Agentic-Function-Calling-Pivot-v1` (9.6k), `Nemotron-Instruction-Following-Chat-v1`
(288k — **sample, do not embed whole**), and/or others named by the operator.
- First test HF reachability with one small set. If HF is unreachable, **STOP and report** —
  this is an infra issue to surface, not to route around.

**GATE:** raw dataset files on disk; per-dataset row counts logged.

### Phase 2 — Select good exemplars (deterministic, low-judgment)
"Good exemplar" is defined operationally — do not use taste:
- **non-empty** text after extraction;
- **token length in [16, 8000]** (fits context; drops degenerate-short and giant items — use
  `NVEmbedAdapter.count_tokens`, which loads only the tokenizer, cheap);
- **deduplicated** (exact hash; near-dup optional);
- **sampled to a target N per dataset** (default 20,000; if a set is smaller, take all),
  sampling **deterministically** — seeded or every-k-th, never random-without-seed.

**Chunk unit (fixed here, not your choice):** to match thought-vault's per-message
granularity, split each conversation into per-message pieces. `chunk_id =
"{dataset_slug}-{example_idx}-msg-{message_idx}"`. Embed the message text (role-tagged text is
fine if consistent across the whole corpus; pick one convention and hold it).

**GATE:** a JSONL with exactly `text` + `chunk_id` per line; N rows; schema-validated; zero
empties; chunk_id scheme consistent.

### Phase 3 — Embed (run the driver)
Run `embed_full_corpus.sh` with your Phase-2 JSONL as `CORPUS`, on the RTX-8000 host.
- Token budgets and `chunk_size` are pre-set for nv-embed; on CUDA OOM the wrapper retries and
  resumes. Monitor with `python scripts/embed_status.py <CKPT>`.

**GATE:** `embed_status.py` reports **done ≥ 99%**; the `<CKPT>.meta.json` reads
`NVEmbed:nvidia/NV-Embed-v2` / 4096 (this is the sidecar proving same model). Log the stuck
residue count.

### Phase 4 — Compact to a clean corpus (the zero-vector guardrail)
The raw checkpoint is append-only: it contains `_failed` zero-vector lines and retry
duplicates. **Do NOT hand the raw file downstream.** Compact with the exact semantics of
`embed_status.py` / `_load_checkpoint`:
- dedupe by `chunk_id`, **keep last write**;
- drop lines with `"_failed": true`;
- drop zero-norm vectors (`dot(v,v) ≤ 1e-10`).
Write `<name>_clean.jsonl`.

**GATE:** clean file has **zero** `_failed`, **zero** zero-norm rows, all `chunk_id` distinct.
Report true completion rate and the stuck-residue count. (For reference, nv4096 itself is
99.71% — 256/87,004 stuck; expect a comparably tiny residue.)

### Phase 5 — Report (deliverable + provenance)
Emit a short meta note beside the clean corpus: dataset(s) + versions, N, chunk unit, model id
(from the model sidecar), generic input JSONL hash/source sidecar identity, true completion rate,
stuck residue, the canonical TVI generation/query provenance used for Phase 0, and the
**Phase-0 invariant cosines** (the proof the path matched). Keep these identities adjacent in the
note but do not collapse their boundaries: TVI owns thought-vault text/IDs; the checkpoint remains
model-keyed vector state; the new external contrast corpus remains a generic JSONL source. Without
that provenance, a future reader cannot trust the comparison.

---

## Why this is a good orchestration test (observability)

The phases are cut so a non-frontier model's orchestration is *legible* — each gate is a place
it either holds or visibly breaks:
- Does it run **Phase 0 before** spending compute (invariant-before-work), or dive straight
  into embedding?
- Does it **STOP on a failed gate** and report, or confabulate success and continue?
- Does it **compact (Phase 4)** or hand the raw zero-laced checkpoint downstream as "done"?
- Does it **verify the sidecar** rather than assume the model was right?
- Does it hold the **chunk unit** fixed, or silently change granularity mid-corpus?

Each is a known failure mode. Watch which gates a given model clears unaided.

## Variants (operator's choice of scope)

- **Minimal (recommended first):** one dataset — `Nemotron-RL-Identity-Following-v1` (21.7k,
  embeddable whole) — single RTX-8000 run. Smallest thing that yields a real contrast leg.
- **Full panel:** several sets for a voice-scale spread. `Cascade-2` (15.9M) must be sampled,
  not embedded whole.
- **Two-workstation split:** the RTX-3090 (24GB) also fits nv-embed fp16, but a *different
  GPU is a hardware variable* — if used, re-run Phase 0 on the 3090 independently before
  trusting its leg. Same-host (RTX-8000) is the invariant-preserving default.

## Downstream (out of scope here; frontier-tier judgment)
Once two clean legs exist, isolating instrument-from-voice (the shared-subspace across corpora
= instrument; each corpus's residual = voice) is a separate, analysis-heavy work-package. This
one only produces the comparable legs.
