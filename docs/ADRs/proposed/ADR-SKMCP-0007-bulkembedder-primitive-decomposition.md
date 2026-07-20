# ADR-SKMCP-0007: BulkEmbedder primitive decomposition — stable facade, frozen artifacts, adopt-style sidecar versioning

**Status:** proposed
**Date:** 2026-07-19 (US/Mountain)
**Related:** ADR-002 (unified adapter — BulkEmbedder's home, `EmbeddingAdapter` contract); ADR-003 (control/data-plane split — bulk ingestion runs *outside* MCP); ADR-SKMCP-0006 (`_failed`/dedup completion semantics the checkpoint contract encodes); sk-mcp#51 (aggregation-semantics commensurability gap, resolved here); ComfyUI-DiffusionGemma#103 (envelope/payload carve, primitive decomposition, extraction fork — the full evidence thread); local-inference-pool#1 (extraction ONE-MINT hygiene cost)

---

## Context

`semantic_kinematics/embeddings/bulk.py` is a working monolith: a single `BulkEmbedder`
class that couples four separable layers — boundary/identity, durability, transport
economics, and embedding payload math (per the decomposition in #103's evidence thread).
Two forces converge on it:

1. **Cross-repo reuse pressure.** ComfyUI-DiffusionGemma's data-boundary work (#103)
   found that the *envelope* primitives BulkEmbedder embodies — mint-identity sidecar,
   self-distrusting resume, typed failure markers, bounded volatile head, ground-verified
   partitioning, budgeted packing — recur across CDG's payload vocabularies (run-log,
   kv_cache serialization, tier-2 DISTRIBUTION, `runs/` banking). Two independent
   implementations already exist (thought-vault embedding bridge; this BulkEmbedder);
   CDG's post-s9 batch would author a third.

2. **A latent commensurability gap (#51).** The `<checkpoint>.meta.json` sidecar guards
   `model_name` + `dimensions` and fail-loud-refuses resume on mismatch. But a stored
   vector's identity also depends on the **aggregation recipe** (L2-normalize-each-then-mean
   direction centroid, normalization order, the antipodal near-zero rejection threshold),
   none of which is carried. A change to that math would mint vectors incommensurable with
   previously banked checkpoints while passing the sidecar guard. Live exposure:
   `design-docs/experiments/idea-corpus-nv4096/vectors.jsonl` is a banked TVI corpus
   artifact whose comparability rests on these semantics staying frozen.

This ADR records the decision complex that lets BulkEmbedder be restructured for reuse
**without** perturbing that banked artifact or its readers. Payload/envelope terminology and
the primitive taxonomy are canon in #103; this ADR does not restate them, it conditions
execution against them.

## Decision

1. **Decompose behind a STABLE FACADE.** BulkEmbedder's public API —
   `__init__(adapter, *, max_tokens_per_request, max_tokens_per_chunk, checkpoint_path,
   prep_window)` and `embed_corpus(items) -> Dict[str, np.ndarray]` — is unchanged by any
   restructuring. Internals may move behind that surface; callers (thought-vault bridge, TVI
   pipeline, `embed_status.compute_status` readers) see no signature or return-shape change.

2. **Artifact contracts are FROZEN byte-compatible.** The on-disk shapes are held bit-stable:
   - Checkpoint JSONL line: `{"chunk_id": ..., "embedding": [...]}` for a success;
     `{"chunk_id": ..., "embedding": <dimensioned-zero-vector>, "_failed": true}` for a
     failure. The `_failed` marker's zero vector is a **fossilized dimensioned-zero-vector
     wart, kept deliberately**: existing readers (`_load_checkpoint`, `embed_status`,
     ADR-SKMCP-0006's dedup) key completion on "right-dimension, non-zero, not `_failed`,"
     so the failure row must carry a `dim`-length zero vector, not `null` or an omitted key,
     or every prior banked checkpoint's failure rows re-parse differently. A cleaner
     zero-free failure schema is rejected here (see Alternatives) precisely because this
     shape is load-bearing for artifacts already on disk.
   - Sidecar `<checkpoint>.meta.json`: JSON object, atomic write-then-rename, `model_name` +
     `dimensions` keys retained.

3. **Checkpoint-load validity becomes an INJECTED VALIDATOR.** The self-distrust primitive
   (re-validate one's own prior artifact line-by-line; corrupt/invalid → retried, never
   trusted) is payload-agnostic and belongs to the durability layer. The specific validity
   test — right dimension, non-zero L2 norm, all-finite (`_is_valid` / the `_load_checkpoint`
   dot-product gate) — is **embedding payload math** and stays embeddings-side, passed *in* to
   the generic loader. The loader owns "distrust and re-check every line"; it does not own
   "what makes a vector valid." This is the envelope/payload carve (#103 s9 ruling 1) applied
   at the seam.

4. **Sidecar gains an aggregation-semantics version field, ADOPT-STYLE (resolves #51).** Add
   a key (e.g. `aggregation_semantics_version`) to `<checkpoint>.meta.json` naming the
   aggregation recipe generation. `_reconcile_meta` today mismatches on any key an old meta
   lacks; the versioned check MUST treat an **absent key as legacy semantics and record it
   forward** — never refuse-on-absent. Concretely: an existing meta without the key is
   adopted as the legacy generation (the same adoption branch a pre-#16 checkpoint already
   takes), the current generation is written on fresh runs, and a *present-and-differing*
   version is the only condition that fails loud. This makes an aggregation-math change a
   detectable mint boundary instead of a silent one, without invalidating a single banked
   artifact.

5. **The GOLDEN-ARTIFACT TEST is the enforcement surface and a landing precondition.** A test
   fixture — a committed fixture checkpoint + fixed input corpus + a deterministic fake
   adapter — asserts that a resume produces a **byte-identical** checkpoint and **bit-identical**
   vectors. This test must exist and pass **before any restructuring lands**. It is what turns
   clauses 1–4 from prose promises into a structural gate: the facade, the frozen bytes, the
   injected validator's equivalence to the inlined one, and the adopt-style version handling
   are each pinned by a diffable golden, not by review vigilance. No restructuring PR merges
   without it green.

Clause 5 is the enforcement surface for clauses 1–2 (contract stability) and for the
extraction fork's adoption gate below. Clause 4 is the enforcement surface for #51.

## Rationale

### Positive Consequences
- **TVI is untouchable by construction.** The banked nv4096 corpus and its readers depend
  only on the frozen bytes (clause 2) and the stable facade (clause 1); the golden test
  (clause 5) fails the moment either drifts. Restructuring cannot reach the artifact.
- **#51 closes without breaking a reader.** Adopt-style versioning (clause 4) records the
  commensurability boundary forward while treating every existing artifact as valid legacy —
  the only correct shape given `_reconcile_meta`'s refuse-on-any-missing-key behavior.
- **The envelope/payload carve gets a code seam, not just a doctrine sentence.** The injected
  validator (clause 3) is where opinion locality becomes structural: durability code carries
  no embedding opinions, so it is reusable by a payload class that has different validity
  (CDG's kv_cache, where magnitude *is* signal).
- **Decomposition without a refactor mandate.** The primitives are *named and seamed*; whether
  they become separate classes/modules in this repo is left to the implementer under the
  golden gate. Local composition may stand.

### Negative Consequences
- **Fossilized wart carried forward, on purpose.** Clause 2 permanently keeps the
  dimensioned-zero-vector failure encoding. It is redundant (the `_failed` flag alone would
  suffice for new readers) and dimension-coupled, but freezing it is cheaper and safer than
  migrating every banked checkpoint. The cost is a documented ugliness in the artifact
  contract that new consumers must honor.
- **A validator injection point is a new coupling surface.** Clause 3 adds a seam that a
  future author could pass a *wrong* validator through (e.g. a norm test with a mismatched
  threshold), reintroducing the exact commensurability hazard #51 names but at the code
  boundary. Mitigated by the golden test pinning the embeddings-side validator's behavior.
- **The version field is only as honest as the author who bumps it.** Clause 4 detects a
  version *mismatch*, not an *un-bumped* aggregation change. An author who alters the centroid
  recipe without incrementing the field defeats the guard. The golden test is the backstop:
  a math change that alters vectors breaks the golden before it can ship silently.

## Extraction Fork — RECORDED, NOT DECIDED

Extraction of the payload-agnostic envelope primitives into a shared, version-pinned library
is a **named fork with a trigger**, not a decision taken here.

- **Trigger:** the third independent implementation of the crossing primitives materializing.
  Two exist (thought-vault embedding bridge, this BulkEmbedder); CDG's post-s9 batch authors
  the third. The fork's decision point arrives **when that batch is dispatched** — not before.
- **Precedent:** `local-inference-pool` — a primitive extracted to its own repo, consumed as a
  version-pinned git dependency (prompt-prix pins `@v0.6.0`). The coordination/drift objection
  is already answered by that mechanism.
- **Library boundary = the envelope/payload carve already ruled (#103 s9):** the extractable
  core is the payload-agnostic layers (identity-sidecar guard, self-distrusting ledger loader
  taking an *injected* validator per clause 3, typed failure markers, bounded head); payload
  math stays repo-local.
- **Cost held honestly:** a new curated repo is a ONE-MINT event (registry entry, born
  replicated) for ~hundreds of lines, and it brings its *own* mint-hygiene tax — LIP#1
  exhibits the failure mode (a README install pin drifting stale against the tag mint, a
  hand-typed version-identity surface that rots each release). Extraction before n=3
  generalizes from n=2.
- **Adoption is ADDITIVE ONLY.** sk-mcp keeps its monolith until it opts in behind the golden
  test (clause 5), or never. CDG's consumers are greenfield post-s9 and may adopt directly; a
  library's arrival imposes nothing on sk-mcp. **TVI is untouchable until sk-mcp opts in**, and
  opting in is itself gated by the golden test proving byte/bit identity across the swap.

## Alternatives Considered

### Option A: Status-quo monolith — do nothing
Keep BulkEmbedder as one class, no facade seam, no version field.
**Rejected.** Fails the n=3 reuse need surfaced in #103: CDG's post-s9 batch would re-implement
the crossing primitives a third time with no shared enforcement surface, and #51's
commensurability gap stays open — a banked-artifact hazard with no detection.
- **Pro:** zero work; no new seams.
- **Con (deciding):** leaves #51 unresolved and lets N re-implementations drift, the exact
  prose-vs-structure gap this repo's doctrine forbids.

### Option B: Immediate extraction to a shared library — now
Extract the envelope primitives to a new repo this pass and consume by pinned dependency.
**Rejected (for now — this is the fork above, not a no).** Generalizing from n=2 mints a repo
prematurely and incurs the ONE-MINT + mint-hygiene cost (LIP#1) before a third consumer proves
the boundary. The trigger is n=3; until then the primitives are named and seamed in place.
- **Pro:** one guard implementation, strongest enforcement.
- **Con (deciding):** premature — n=2 does not yet justify a repo mint and its per-release
  hygiene tax; the boundary is better proven by CDG's third implementation first.

### Option C: Clean generic-ledger schema — redesign the artifact
Replace the fossilized checkpoint shape (dimensioned-zero failure row, embedding-specific
keys) with a payload-agnostic generic ledger format.
**Rejected.** Breaks every existing reader (`_load_checkpoint`, `embed_status`,
ADR-SKMCP-0006's dedup) and every banked TVI artifact
(`design-docs/experiments/idea-corpus-nv4096/vectors.jsonl` + sidecar). The comparability of a
frozen corpus is exactly what must not move; a schema clean-up is not worth invalidating the
banked product.
- **Pro:** a tidier, dimension-decoupled artifact for new consumers.
- **Con (deciding):** breaks readers and the banked corpus — a corpus-stability violation for
  cosmetic gain.

## Anticipated-Failure Register

Per the greenfield-adaptation discipline (each invariant names the failure it prevents,
before that failure occurs):

| Invariant (clause) | Failure it prevents |
|---|---|
| Stable facade (1) | A restructuring silently changes `embed_corpus`'s signature/return shape, breaking the thought-vault bridge or TVI pipeline caller mid-run. |
| Frozen bytes incl. zero-vector wart (2) | A "cleanup" drops or `null`s the failure row's zero vector; every banked checkpoint's `_failed` rows re-parse as completed (or crash the loader), corrupting completion accounting. |
| Injected validator (3) | Durability code grows an embedding opinion (norm/dim), then gets reused for a payload where magnitude is signal (kv_cache), silently rejecting valid data. |
| Adopt-style version field (4) | A refuse-on-absent version check bricks resume on every pre-existing checkpoint; or an absent field is treated as "current," masking a real semantics change. |
| Golden-artifact test (5) | An aggregation-math change ships without a version bump and mints incommensurable vectors into the banked corpus undetected. |

## Open Questions

- [ ] **Field name and value vocabulary for `aggregation_semantics_version`.** Monotonic
  integer generation vs. a content hash of the recipe constants (normalization order,
  antipodal threshold, centroid method). **Resolution trigger:** decided by the first
  implementer of clause 4 under the golden test; a hash is preferable if it can be derived
  from the code path rather than hand-bumped (removes the "author forgot to bump" failure in
  the Negative Consequences), but only if that derivation is cheap and stable.
- [ ] **Whether the injected validator is a callable or a small protocol object.** Affects how
  cleanly the extracted library (if the fork fires) accepts CDG's payload validators.
  **Resolution trigger:** settled at extraction time (fork above), when a second real
  validator (kv_cache's magnitude-preserving check) exists to generalize against — not before,
  to avoid designing the seam from n=1.
- [ ] **Does the golden fixture pin the antipodal-collapse path?** The near-zero-mean rejection
  (`_l2_normalize(mean)` on cancelled sub-chunks → `_is_valid` fail → `_failed`) is part of the
  aggregation semantics #51 guards. **Resolution trigger:** the golden corpus should include at
  least one multi-sub-chunk item exercising the collapse branch, decided when the fixture is
  authored (clause 5 precondition).

## Supersession Relationships

**Supersedes:** — (no prior ADR; extends ADR-002's BulkEmbedder and ADR-SKMCP-0006's
completion semantics without replacing either).
**Superseded by:** TBD.

## Implementation Notes

No implementation lands under this ADR. Restructuring, the injected-validator seam, the
sidecar version field, and the golden-artifact test are follow-on work gated by clause 5
(golden test green before any restructuring merges). The extraction fork is dormant until its
n=3 trigger fires (CDG post-s9 batch). Commentary and the full evidence chain live by pointer
in sk-mcp#51 and ComfyUI-DiffusionGemma#103.
