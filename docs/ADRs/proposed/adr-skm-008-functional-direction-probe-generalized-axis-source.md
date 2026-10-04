# ADR-SKM-008: Functional-direction probe — seedset centroids as a generalized axis source over the existing projection contract

**Status:** accepted (operator-ratified 2026-07-29)
**Date:** 2026-07-29 (US/Mountain)
**Author:** shanevcantwell, with Claude (Opus, design subagent) as drafting collaborator
**Related:**
- **Ecosystem law** — `operating-doctrine/ground-physics/CODE_CONSTITUTION.md` + `GROUND_PHYSICS.md` (the data-plane identity invariants and the "every position is a contract" recursion this design instantiates, not a house style it conforms to); `docs/ARCHITECTURE.md` (the four-rule layering invariant — control-plane core / shared substrate / contract / consumer plane).
- **Direct lineage this ADR extends** — ADR-001 (referential axis-alignment: the empirical-null / z-score-against-a-model-keyed-manifest discipline; `build_axis_null.py` → `analyze_axis_alignment` build-then-consume template); **ADR-SKMCP-0001** (the directional-projection *math*: signed component + orthogonal residual + cosine, measured null, single-embedder — this ADR adds a *new axis source* to that primitive, it does not mint a second primitive); **ADR-SKMCP-0002** (the *tool-family contract*: `initialize`-builds-validated-artifact / `run`-consumes-it, embedder pinned by the artifact, cross-cutting rules encoded as refusals — this ADR reuses that contract shape and generalizes its axis-source slot).
- **Calibration result productized here** — `scripts/measure_cone.py` + `docs/runbooks/anisotropy-instruments.md` §4 (‖μ‖=0.555; centered participation-ratio ≈34 vs uncentered ≈8; "persist μ + eigenbasis as a model-keyed calibration artifact" named as future work — this ADR does it).
- **Regime-typed artifact discipline reused** — `semantic_kinematics/bearing/jolt.py` (`load_null` hard-fail on missing/under-described/wrong-regime header — the self-describing-artifact loader pattern applied to the new artifacts).
- **Frozen input contract (sibling repo)** — `thought-vault-integration/docs/adrs/adr-tvi-008-longitudinal-construction-tracking-corpus-side.md` §Boundary contract: `corpus.db`, `vectors_nv-embed-v2.f32` + `corpus_join_manifest.json`, `<pattern_id>.seedset.json`. Files-only boundary; refuse on `vector_memmap_sha256` / `embedding_model_id` mismatch. The corpus meta id `NVEmbed:nvidia/NV-Embed-v2` is a known re-typed non-canonical id (tvi#41 class) — **flagged and coordinated, not propagated and not solved here**. (Note: this sibling ADR was renumbered from ADR-TVI-007 to ADR-TVI-008 on 2026-07-29 — first-landed `adr-tvi-007-training-data-pipeline.md` held the 007 mint under `ONE-MINT`; all references here updated accordingly.)
- **Program framing** — `docs/SPINE.md` (Aim 3: longitudinal model-behavior drift across generations — this is the instrument for it); `docs/HANDOFF.md` §8 (the position/bearing regime split this design sits alongside).

**Supersedes:** — (additive to the sk-mcp contracted surface; supersedes no tool).
**Superseded by:** TBD.
**Rename note:** drafted as `ADR-SKMCP-0007`; renamed to `ADR-SKM-008` on acceptance (2026-07-29) — the registry (`design-docs/adr-refactor/codes.txt`) rules `SKM` the canonical repo code for semantic-kinematics-mcp, `SKMCP` a flagged dangling mint (reconcile-pending for ADR-SKMCP-0001–0006). Mirrors commit `b912ae9`'s prior correction of the bulkembedder ADR to `adr-skm-007`, which also means 007 was already occupied — this ADR takes 008, the next free `SKM` number.

---

## Context

The operator wants to chase rhetorical *constructions* whose **function** is near-orthogonal to topic — comparative-perception praise ("better than it looks"), prior-violation praise ("bizarrely consistent") — through the 127K-chunk chat corpus, **longitudinally across assistant-model eras** (Claude 4.6/4.7/4.8, Gemini), and read how per-construction rates shift in composition across eras (the operating hypothesis: under RLHF constraint shift, per-construction rates re-compose while the aggregate holds level). This is the concrete instrument for SPINE Aim 3.

Three facts fix the design's shape, and each cites the law it descends from:

1. **This is a new *contracted position* in the ecosystem, not a product built in sk-mcp's local style.** GROUND_PHYSICS makes "every position is a contract, not a tool" recursive — it binds at design depth. The corpus side is tvi (data plane, ADR-TVI-008); sk-mcp is the contracted analysis control plane (ARCHITECTURE.md Rule 2: MCP is the sole door); llauncher is the lifecycle plane; the operator's local qwen orchestrators and the (future) Gradio UI are the consumer plane. The primitives designed here **enter the existing MCP surface** through `mcp/server.py`'s `list_tools()`/`call_tool()` (ARCHITECTURE.md "instant fail" table: a tool in `commands/` with no `server.py` entry, or a consumer bypassing the contract, is a violation). They are governed by the same four rules as every other primitive; nothing below is a repo convenience.

2. **A seedset-derived direction is a *generalized axis source*, not a new instrument.** ADR-SKMCP-0001 fixed the projection math and named the axis as its pluggable input: "a reference axis defined by two anchor texts whose difference *is* the axis." ADR-SKMCP-0002 shipped that primitive as a build-validated-artifact + one-call-consume family and stated the axis source is exactly the swappable part ("the axis comes from a curated anchor grid, not the vault"). **A functional direction is that same axis with its poles computed from corpus-derived, topic-matched *centroids* instead of typed exemplar text.** Reusing one projection machinery with two axis sources (`typed_exemplars` | `seedset_centroids`) is what the constitution's `ONE-DOOR` and the ADR-0001/0002 lineage demand; minting a parallel "direction-probe instrument" beside `analyze_axis_alignment` / the bearing family would be the fossil-duplication those ADRs exist to prevent. Where this design keeps anything *separate* from the existing machinery (§Decision D2, D4), the ADR justifies why generalization fails there specifically.

3. **The corpus-scale work is interactive-speed once TVI-008's substrate exists — so it is control-plane, not batch.** An earlier framing pushed direction extraction, projection, and rate-tables into data-plane CLI scripts on the assumption "corpus-scale ⇒ long-running batch ⇒ out of MCP." That assumption is **void** against the real substrate: with `corpus.db` + the `N×4096` float32 memmap already built by tvi, a difference-of-centroids over a few hundred seed rows is milliseconds, and a full-corpus projection is one `127K×4096 · 4096` matvec — seconds, ≈2 GB read, no GPU, no embedding-model call. ARCHITECTURE.md keeps *bulk embedding* (the data-plane job producing vectors) outside MCP; it does **not** exile cheap linear algebra over already-computed vectors — that is exactly the "precomputed-vector consume is not bulk ingestion" carve-out ADR-SKMCP-0002 §Decision already relied on. The **only** residual data-plane script in this ADR is the μ/eigenbasis calibration build (one `4096×4096` `eigh`, minutes) — mirroring that `build_axis_null.py` is a script while `analyze_axis_alignment` is a tool.

### What exists to build on (grounded)

- `mcp/server.py:33-42` (`list_tools()` extends per command module) and `:45-83` (`call_tool()` dispatch) — the registration seam every new tool joins.
- `mcp/commands/axis_alignment.py` — `alignment_core` (pure, IO-free, exhaustively unit-testable; `docs/axis-alignment.md` specifies it), `build_null_cache`/`load_null_cache` (model-keyed manifest + refuse-on-mismatch). The projection kernel and the manifest discipline are here already.
- `bearing/jolt.py:109-165` (`load_null`) — the regime-typed self-describing loader: required header keys, hard-fail on missing/wrong regime, no silent default, no legacy reader. The artifact loaders below follow this exactly.
- `scripts/measure_cone.py:176-211` — μ, ‖μ‖, and both eigenspectra already computed with the exact dedup/zero-vector convention; §Decision D1's calibration build lifts this numeric core (per the runbook §4.2 "factor a pure numeric core" instruction) rather than re-deriving it.

---

## Decision

Add **one axis-source generalization** and **one calibration artifact** to the existing contracted surface, then expose the whole chase-down loop as **stateless MCP primitives** in `mcp/commands/`, registered in `server.py`. Durable artifacts (μ/eigenbasis, functional directions, projection+rate tables) keep the ADR-SKMCP-0002 build-validated / one-call-consume shape, are keyed by **canonical embedding-model identity**, and — wherever interactive speed permits — are **minted through primitives**, not scripts. The lone script is the calibration build.

### D0 — Mount point: a new command module `mcp/commands/direction.py`, registered like every other

`server.py:37-41` gains `tools.extend(direction.get_tools())`; `call_tool()` gains the dispatch arm for each verb below. This is the enforcement surface for ARCHITECTURE.md Rule 2 — no tool reachable except through `server.py`. All handlers are `async def handler(state_manager, arguments) -> dict`, stateless (resolve-use-release; no cross-call retained state), taking **artifact refs and file paths** as parameters, returning `{...}` or `{"error": "..."}` with an *instructive* message (ADR-SKMCP-0002 "errors instruct").

### D1 — μ/eigenbasis calibration artifact (**the one data-plane script**)

`scripts/build_corpus_calibration.py` (argparse; structured JSON to stdout, progress to stderr — the prompt-prix ADR-007 CLI shape the operator's qwen drives). It reads the TVI-008 **memmap** (not the JSONL — the memmap is the frozen dense contract) + `corpus_join_manifest.json`, and emits a self-describing, model-keyed calibration artifact under `data/calibration/`:

```
data/calibration/<embedding_model_slug>.calibration.npz   # mu (4096,), optionally eigvecs (4096,k), eigvals (k,)
data/calibration/<embedding_model_slug>.calibration.json  # manifest (below)
```

Manifest (regime-typed, jolt.py-style required-header discipline):

```json
{
  "header": {
    "regime": "corpus-calibration",
    "embedding_model_id": "nvidia/NV-Embed-v2",        // CANONICAL — sourced from the tvi manifest, NOT re-typed
    "embedding_model_id_source": "tvi corpus_join_manifest.json",
    "embedding_model_id_flag": "tvi meta sidecar carries re-typed 'NVEmbed:nvidia/NV-Embed-v2' (tvi#41 class); canonical form used here, mismatch surfaced not propagated",
    "dim": 4096,
    "source_memmap_path": "…/vectors_nv-embed-v2.f32",
    "source_memmap_sha256": "…",                        // the identity gate every downstream artifact inherits
    "n_used": 126729,
    "convention_version": "tvi-008-dedup-keep-last-zerofilter-v1"
  },
  "mu_norm": 0.555,
  "eigenbasis_included": true,
  "eigenbasis_k": 256,
  "participation_ratio_centered": 34.1,
  "built_at": "2026-07-29T…Z"
}
```

**Decision: persist μ *and* a truncated eigenbasis (top-k, k≈256), not μ alone.** Cost is one `4096×4096` `eigh` (minutes, already run by `measure_cone.py`); the eigenbasis is what a future SBR "personal-geometry state vector" (runbook §4.5) and any whitening/denoising diagnostic need, and computing it now avoids a second full pass later. It is **optional at load** (a consumer that only mean-centers reads `mu`; the eigenbasis is inert until used) so its presence costs downstream nothing.

**Why μ is a persisted artifact and not recomputed per call:** the corpus mean is a fixed property of the frozen corpus+embedder; recomputing it per projection would re-stream 2 GB every call. Persisting it keyed by `source_memmap_sha256` makes mean-centering O(1)-to-load and makes staleness *detectable* (a rebuilt memmap changes the sha; every direction/projection artifact records which μ-sha it centered against and refuses a mismatch — the ADR-001 model-keyed-refusal pattern applied to the calibration layer).

### D2 — Direction extraction: a **generalized axis source** on the existing projection kernel (minted through a primitive)

`initialize_direction` (MCP tool, patient budget). Input: a **seedset artifact ref** (TVI-008 `<pattern_id>.seedset.json`) + a **calibration ref** (D1). It:

1. **Refuses on identity mismatch first** (ARCHITECTURE.md Rule + ADR-001 discipline): the seedset manifest's `corpus_snapshot.embedding_model_id` and `vector_memmap_sha256` must match the calibration's `embedding_model_id` and `source_memmap_sha256`. A mismatch means the seeds index a different vector population than μ was computed over — refuse, name the mismatch. This is the files-only boundary's integrity gate (TVI-008 §Boundary "sk-mcp refuses on mismatch").
2. **Reads seed + negative rows from the memmap** by `rowid_mm` (denormalized into the seedset — no `corpus.db` query needed, no cross-repo import). Casts float32, mean-centers each by subtracting μ (D1). **Mean-centering is sk-mcp's job** (TVI-008 ships raw unit-normalized vectors; centering here is the ‖μ‖=0.555 anisotropy removal that turns a difference vector into a *functional* direction rather than a restatement of the common mean).
3. **Computes the axis via the existing kernel's contract**: `direction = centroid(centered seeds) − centroid(centered topic-matched negatives)`, then unit-normalize. This is `build_axis`'s difference-of-poles (`docs/axis-alignment.md` `build_axis`) with the poles supplied as **centroids over corpus rows** instead of **means over embedded exemplar text**. *Same operation, new source.* `pole_separation = ‖raw‖` is reported and gated exactly as `alignment_core` gates it.
4. **Era-scoped extraction (first-class, for cross-projection):** an optional `era` filter selects only seeds/negatives whose denormalized `era` matches, yielding a per-era direction `d_A`. With no filter, the global direction. The tool emits **one direction artifact per (pattern_id, era-scope)**.
5. **Emits a self-describing direction artifact** (`data/directions/<pattern_id>[.<era>].direction.npz` + `.json`), regime `functional-direction`, carrying the **full provenance chain**: seedset manifest (pattern_id, regex_sha256, corpus_snapshot), calibration manifest (μ-sha, embedding_model_id), era-scope, counts, `pole_separation`, and the validation diagnostics of D3.

**Why this is a generalization, not a parallel primitive:** the projected quantity, the null discipline (D3/D5), the artifact shape, and the refuse-on-mismatch loader are all ADR-0001/0002's. The only new thing is *the axis source is a corpus-centroid difference*. Adding it as a second `axis_source` value keeps one door. — **Where generalization is bounded (honest):** the *seedset ingestion + centroid + mean-centering* step is genuinely new work `analyze_axis_alignment` does not do (it embeds anchor text at call time; here the vectors are precomputed and centered against a persisted μ). So `initialize_direction` is a distinct *builder verb* — but it emits the **same artifact class** the projection verbs already consume. The generalization is at the artifact/projection layer; the builder differs because its *input is precomputed corpus rows*, not typed text. That difference is inherent (the whole point is topic-matched corpus negatives), so a shared builder would fail — this is where separation is justified.

### D3 — Validation diagnostics baked into the direction artifact (the honesty surface)

Per GROUND_PHYSICS/ADR-SKMCP-0002 rule 4 (falsification-shaped confidence: a too-clean result is an *alarm*), `initialize_direction` computes and records, on **held-out** splits:

- **Held-out separation AUC** — split seeds/negatives (e.g. 5-fold or 70/30 by `paired` groups so a seed and its matched negative never straddle the split), extract the direction on train, project held-out seeds vs held-out negatives, report **AUC** of the projection as a separator. This is the "does the direction actually distinguish the construction from its topic-matched control" test. **A suspiciously high AUC (e.g. >0.98) is surfaced as a circularity/leakage alarm, not a win** (ADR-SKMCP-0002 rule 4; the regex is a near-perfect labeler, so leakage is a live risk).
- **Topic-control check** — project the held-out **negatives** and a random topic-matched sample onto the direction; if the negatives (same-conversation, same-speaker, no construction) separate from the *seeds* but a random-topic sample does *not* shift, the direction is functional not topical. Report the negative-vs-random separation as the topic-orthogonality evidence. (This is the corpus-side topic-matching from TVI-008 §Context cashing out as a measurable check here.)
- **Bootstrap stability of the direction** — resample seeds/negatives with replacement B times (e.g. B=200), re-extract, report the **mean pairwise cosine of the bootstrapped directions** (direction reproducibility) and a bootstrap CI on `pole_separation`. A direction whose bootstrap cosine is low is under-determined by too few seeds — refuse to promote it (parallels `alignment_core`'s `min_pole_separation` gate, lifted to sampling stability).
- **Null calibration reference** — the projection z-score null is the **mean-centered corpus projected onto this direction** (`μ₀≈0` by construction after centering; `σ₀` = corpus spread along the axis), exactly ADR-001 `null_stats` but over the memmap population rather than a separate null corpus. Recorded so `run`/rate verbs z-score without recomputation.

These are chosen over alternatives (per the dispatch's "your choice, justified"): AUC because the regex gives a clean binary label so a ranking metric is honest and threshold-free; topic-control because it directly tests the *near-orthogonal-to-topic* claim that is the instrument's whole reason to exist; bootstrap because seed counts per (pattern×era) will be small and stability is the failure mode that silently produces confident noise. Distribution-shape of the projection is also recorded (ADR-SKMCP-0002 OQ: if non-Gaussian, rate thresholds use empirical quantiles not σ).

### D4 — The chase-down loop as contracted MCP primitives (the interface the design serves)

All stateless, artifact-ref parameters, loud refusal on identity mismatch, registered in `server.py`. Naming follows the ADR-SKMCP-0002 lifecycle convention so a non-frontier agent reads the workflow off the tool list:

| Tool | Role | Budget | Params (artifact refs / paths) |
|---|---|---|---|
| `initialize_direction` | D2/D3: seedset + calibration → mean-centered difference-of-centroids direction + validation diagnostics; era-scoped. Emits direction artifact. | patient | `{seedset_ref, calibration_ref, era?}` |
| `project_text` | Embed input text (single-embedder, pinned by the direction artifact's `embedding_model_id`), mean-center against the artifact's μ, project onto the direction → `{projection, z, cosine}`. The interactive "does *this* passage carry the construction" probe. | fast | `{text, direction_ref}` |
| `project_chunks` | Project a caller-supplied list of `rowid_mm` (or `chunk_id`) from the memmap onto the direction → per-row `{rowid_mm, z}`. No embedding call — pure memmap read + matvec. | fast | `{direction_ref, rowids?|chunk_ids?}` |
| `project_corpus` | Full-corpus projection: the `N×4096 · 4096` matvec over the whole memmap → persists a **projection artifact** (`data/projections/<pattern_id>[.<era>].proj.npz`: per-`rowid_mm` z) keyed by direction-sha + memmap-sha. Seconds; minted through the primitive (interactive-speed), not a script. Returns the artifact ref + summary stats. | patient (seconds) | `{direction_ref}` |
| `query_rates` | Read a projection artifact + `corpus.db` (read-only, for `era`/`channel`/`speaker` grouping) → aggregated **rate table**: fraction of chunks above the calibrated threshold, by `era × channel × speaker`. Threshold from D5. Emits/updates a rate-table artifact. | fast | `{projection_ref, threshold?, group_by?}` |
| `cross_project` | The **era-composition matrix**: given a set of per-era direction artifacts (extracted from era A's seeds) and projection over era B's corpus slice, build the `era-direction × era-corpus` matrix of rates. This is the hypothesis test — direction from era A applied to era B. | patient (seconds) | `{direction_refs[], projection_ref|corpus scope}` |
| `top_exemplars` | Top-K `chunk_id`s along a direction (from a projection artifact), **with text fetched via `corpus.db`** (read-only) for readback. The operator's "show me what scored high" chase primitive. | fast | `{projection_ref|direction_ref, k, era?}` |
| `direction_diagnostics` | Read back a direction artifact's D3 diagnostics (AUC, topic-control, bootstrap, verdict) without re-running — the "should I trust this axis" inspect verb. Parallels `get_bearing_analysis_status` / `model_status`. | instant | `{direction_ref}` |

**Audited override (implemented, PR [#65](https://github.com/shanevcantwell/semantic-kinematics-mcp/pull/65)):** `project_text`, `project_chunks`, and `project_corpus` each accept an additional `allow_non_usable_direction` (default `false`) param that lets the caller proceed against a non-`usable` verdict; when exercised, the result payload is stamped `allow_non_usable_direction_used: true` — refuse-by-default is preserved as the default and every use of the escape hatch is self-recorded on the result, never laundered silently (`semantic_kinematics/mcp/commands/direction.py:995-1010, 2035, 2071` on `origin/main`).

**Consumer note (D4 binds the interface):** the operator's local qwen orchestrators drive this whole loop over MCP — that is the interface the design exists to serve. A Gradio tab is a *future* consumer (out of scope; **nothing above forecloses it** — every capability is a contracted primitive a UI would call, never UI-specific core logic, per ARCHITECTURE.md Rule 3). A CLI, if added, is a **thin wrapper over these same primitives** (prompt-prix ADR-007 shape), never a capability of its own.

### D5 — Threshold / z-score calibration (the null is the experiment)

Rates in `query_rates`/`cross_project` require a threshold on the projection. **Decision: the threshold is calibrated against the regex labels via the D3 held-out AUC curve, and the reported significance is a z-score against the mean-centered-corpus null (D3), not an absolute projection.** Concretely: `initialize_direction` records the projection distribution of held-out seeds and of the corpus null; the operating threshold is the value on the corpus-null projection axis that achieves a chosen precision against the regex-labeled held-out seeds (default: the projection z at which held-out seed-vs-negative precision ≥ 0.9, since the regex is a near-perfect-precision labeler — ADR-001 "the null is the experiment," made corpus-relative). The threshold and the precision it was set at are **recorded in the rate-table artifact's manifest** so every rate is re-derivable and self-labeling (ADR-SKMCP-0006: derived stats are generated, never hand-typed). If the projection distribution is non-Gaussian (D3 shape check), the threshold is an **empirical quantile**, not a σ-multiple.

### D6 — Optional interactive pattern-preview (boundary decision, stated either way)

**Decision: include a read-only `preview_pattern` primitive; do NOT mint seed sets here.** Seed-set *minting* stays corpus-side (TVI-008 §Component 4 owns negative-pairing, which needs conversation adjacency + speaker + text — corpus-side facts; ADR-TVI-008 Option A rejects pushing pairing to sk-mcp for exactly this reason). But an interactive `preview_pattern {regex, k}` that opens `corpus.db` **read-only** and returns the first K matching `chunk_id`s + text — *without* building negatives or an artifact — lets the operator's orchestrator iterate on a regex before asking tvi to mint a full seedset. It reads, never writes, never crosses into pairing; the boundary (minting is corpus-side, previewing is a read) is preserved. Rejected the alternative of *no* preview: it would force a full tvi seedset-mint round-trip for every regex tweak, and the read is cheap and boundary-safe.

---

## Rationale

The pattern is proven twice (ADR-001 position regime, ADR-SKMCP-0002 bearing regime); this is the same build-validated-artifact + one-call-consume shape with a **third axis source** (corpus-centroid difference) mounted on the **one** projection door. The layering is sanctioned: cheap linear algebra over already-computed vectors is control-plane consume, not data-plane bulk ingestion — the exact carve-out ADR-SKMCP-0002 §Decision already established. Every load-bearing invariant maps to a named enforcement surface (§below), because a rule enforced only by prose is one refactor from gone (CODE_CONSTITUTION "name the enforcement surface").

### Enforcement surfaces (per invariant touched)

| Invariant (handle) | How satisfied here | Enforcement surface |
|---|---|---|
| `ONE-DOOR` | Seedset direction is a new axis *source*, not a second projection instrument; the projection kernel, null discipline, and artifact loader are ADR-0001/0002's. | code review: `initialize_direction` reuses `alignment_core`'s `build_axis`/`null_stats` math (or a shared kernel factored from it); no second projection implementation in `direction.py`. A test asserts a `typed_exemplars` axis and a `seedset_centroids` axis produce identical projections given identical pole vectors. |
| ARCHITECTURE Rule 2 (MCP sole door) | Every verb is registered in `server.py` `list_tools()`/`call_tool()`; no consumer imports `direction.py` directly. | a test asserts every `direction.get_tools()` entry has a `call_tool` dispatch arm (the "no orphan tool" check ARCHITECTURE.md's instant-fail table names). |
| ARCHITECTURE Rule 1 (stateless core) | Handlers resolve-use-release; artifacts are refs on disk, no cross-call state. | same-in/same-out test over `project_text`/`project_chunks`; no module-level mutable state in `direction.py`. |
| `CONSERVE-DATA-BOUNDARY` / identity gate | Every artifact records `embedding_model_id` + `source_memmap_sha256`; every consuming verb refuses on mismatch (calibration↔seedset↔direction↔projection chain). | the regime-typed loader (jolt.py `load_null` pattern) hard-fails on missing/mismatched header; a round-trip test builds→loads→asserts sha-refusal fires on a mutated sha. |
| `EMIT-CANONICAL` / `ONE-MINT` (coordination) | `embedding_model_id` is the **canonical** `nvidia/NV-Embed-v2`, sourced from tvi's `corpus_join_manifest.json`; the re-typed `NVEmbed:nvidia/NV-Embed-v2` (tvi#41 class) is **flagged in the manifest, never propagated**; no new re-typing minted. | a grep-gate forbidding `NVEmbed:` (or other prefixed/`.gguf`) id strings in artifact writers; the manifest's `embedding_model_id_flag` field records the coordination; refuse-at-ingress on an unknown id (no `"unknown"` fallback). |
| Falsification-shaped confidence (GROUND_PHYSICS / ADR-0002 r4) | D3 held-out AUC + circularity alarm on >0.98; bootstrap stability gate; too-clean is an alarm not a win. | `initialize_direction` returns `verdict` ∈ {`usable`, `under-determined`, `leakage-suspected`} recorded in the artifact; `project_*` refuse a non-`usable` direction with an instructive error. |
| Derived-stats-generated (ADR-SKMCP-0006) | μ, directions, projections, rate tables are all generated artifacts with self-labeling manifests; no hand-typed rate ever enters a doc. | the manifests carry `convention_version` + source shas + `built_at`; rate tables are regenerated, not authored. |

### Positive consequences

- One projection door gains a corpus-native axis source; the operator's qwen can drive extraction→projection→rates→exemplar-readback→cross-projection entirely over MCP, no frontier model in the deterministic loop (prompt-prix pattern).
- Aim 3 becomes a runnable, falsifiable instrument: per-era direction extraction + the `cross_project` era-composition matrix is exactly the "rates re-compose while aggregate holds level" hypothesis, testable with a null and held-out validation.
- μ/eigenbasis calibration (runbook §4.5's named future work) lands as a reusable, model-keyed artifact — the SBR "personal geometry" state vector inherits it for free.

### Negative consequences

- A new command module + eight verbs is a larger surface than the bearing family's three; mitigated by the shared `direction`/`project_` stems and lifecycle verbs, and by `direction_diagnostics`/`preview_pattern` being the droppable-for-v1 pieces.
- The direction artifact denormalizes μ-sha and memmap-sha; a rebuilt corpus invalidates directions — but *detectably* (refuse-on-mismatch), never silently.
- `initialize_direction` owns real statistical work (held-out AUC, bootstrap) — implementation cost justified only by the patient-setup budget (ADR-SKMCP-0002 asymmetric-latency argument).
- Threshold calibration against a near-perfect regex labeler risks measuring the regex, not the function; D3's topic-control + leakage alarm is the guard, but this is the design's sharpest honesty risk and is called out as such.

## Alternatives Considered

### Option A: Mint a standalone "direction-probe instrument" beside `analyze_axis_alignment` / the bearing family
**Rejected.** It duplicates the projection kernel, the measured-null discipline, the model-keyed-artifact loader, and the refuse-on-mismatch machinery that ADR-0001/0002 already contract — the exact fossil-duplication those ADRs exist to prevent (ARCHITECTURE.md ONE-DOOR). The seedset direction differs from a typed-exemplar axis *only in its source*; that is an axis-source parameter, not a new instrument. Generalization succeeds here, so separation is unjustified.

### Option B: Direction extraction / projection / rates as data-plane CLI scripts (the earlier framing)
**Rejected — the false-batch assumption is void.** Difference-of-centroids over hundreds of seed rows is milliseconds; full-corpus projection is a seconds-scale `127K×4096` matvec over the precomputed memmap. ARCHITECTURE.md exiles *bulk embedding* (GPU, model calls) from MCP, not cheap linear algebra over already-computed vectors — the "precomputed-vector consume is not bulk ingestion" carve-out ADR-SKMCP-0002 relied on. Scripting it would put the operator's orchestrator outside the tool surface for the core loop, breaking "teach the workflow from the tool list." Only the μ/eigenbasis `eigh` (minutes) stays a script, mirroring `build_axis_null.py`.

### Option C: Mint seed sets in sk-mcp (fold the labeler here)
**Rejected.** Negative-pairing needs conversation adjacency + speaker + text — corpus-side facts owned by tvi (ADR-TVI-008 §Component 4 / Option A). Pulling it here re-couples sk-mcp to tvi's chunk layout across the files-only boundary. Seed minting stays corpus-side; sk-mcp gets only the read-only `preview_pattern` (D6) for regex iteration.

### Option D: Persist μ only, recompute the eigenbasis when needed
**Rejected.** The `eigh` is already computed in the same pass `measure_cone.py` runs for μ; deferring it means a second full corpus read later for the SBR/whitening consumers. Persisting the top-k eigenbasis now (optional at load, inert until used) costs one artifact write and no downstream burden.

### Option E: Cross-repo Python import of tvi's seedset builder / corpus view
**Rejected — inherited from TVI-008.** The boundary is files-only (memmap + manifest + seedset JSON); sk-mcp refuses on sha/id mismatch and never imports tvi. This keeps the measurement side runnable against a frozen snapshot, matching the stateless-core / null-cache discipline.

## Phased roadmap

Dependency-respecting; each phase independently verifiable with named test coverage. If per-sub-problem enumeration with acceptance criteria is needed before dispatch, run a `decompose-problem` pass over Phase 2/3 (the widest) — not reproduced here.

- **Phase 1 — Calibration artifact (the one script).** `scripts/build_corpus_calibration.py`: read TVI-008 memmap + manifest → μ (+ top-k eigenbasis) → model-keyed `.calibration.{npz,json}`; a `load_calibration` loader in `direction.py` following jolt.py `load_null` (hard-fail on missing/wrong-regime/missing-key). *Delivers:* the mean-centering + null-reference substrate. *Depends on:* TVI-008 Phase 1 (memmap + manifest exist). *Verifiable:* `mu_norm` reconciles to ≈0.555; artifact sha-refusal fires on a mutated memmap sha; loader rejects a header missing `embedding_model_id`. *Touches:* new `scripts/build_corpus_calibration.py` (lifts `measure_cone.py:176-211` numeric core), new `direction.py::load_calibration`, `tests/test_calibration.py`.

- **Phase 2 — `initialize_direction` + validation diagnostics (the generalization).** Seedset+calibration → mean-centered difference-of-centroids over `build_axis`; D3 diagnostics (held-out AUC, topic-control, bootstrap, null reference, verdict); era-scoped extraction; direction artifact. *Delivers:* the functional direction as a generalized axis source, validated. *Depends on:* Phase 1 + a TVI-008 seedset artifact (Phase 4 there). *Verifiable:* identity-mismatch (seedset id ≠ calibration id) refuses; a `typed_exemplars` axis and a `seedset_centroids` axis with identical poles project identically (ONE-DOOR test); an injected leakage case trips the `leakage-suspected` verdict; bootstrap-cosine on a too-thin era refuses. *Touches:* `mcp/commands/direction.py` (+ shared kernel factored from `axis_alignment.alignment_core`), `server.py:37-41,45-83` (register), `tests/test_direction_extraction.py`.

- **Phase 3 — Projection + rate primitives.** `project_text`, `project_chunks`, `project_corpus` (persist projection artifact), `query_rates` + D5 threshold calibration, `top_exemplars` (corpus.db text readback). *Delivers:* the interactive chase loop + longitudinal rate tables. *Depends on:* Phase 2 (direction artifacts). *Verifiable:* `project_corpus` matvec reconciles row count to the memmap `n_used`; a rate table's manifest records threshold + precision + source shas (regen-not-authored); `top_exemplars` fetches text by `chunk_id` via corpus.db read-only; `project_text` refuses a non-`usable` direction. *Touches:* `direction.py` (verbs), `server.py`, `tests/test_projection_rates.py`.

- **Phase 4 — Cross-projection matrix + inspect/preview verbs.** `cross_project` (era-direction × era-corpus matrix), `direction_diagnostics` readback, `preview_pattern` (corpus.db read-only). *Delivers:* the Aim-3 hypothesis test surface + regex-iteration ergonomics. *Depends on:* Phase 2 (per-era directions) + Phase 3 (projections). *Verifiable:* the matrix's off-diagonal (era-A direction on era-B corpus) is populated and each cell records both source directions; `preview_pattern` writes nothing (read-only assertion); `direction_diagnostics` returns the recorded verdict without recomputation. *Touches:* `direction.py`, `server.py`, `tests/test_cross_projection.py`.

## Risk and observability

- **Regex-labeler leakage (primary honesty risk).** The near-perfect regex labeler can make the direction *measure the regex*, not the function. **Enforcement:** D3 held-out AUC with a >0.98 circularity alarm; topic-control check (negatives-vs-random must not shift while seeds-vs-negatives do); `verdict` gate blocks `project_*` on `leakage-suspected`. A too-clean result is an alarm, not a win (ADR-SKMCP-0002 r4).
- **Identity drift across the artifact chain.** μ, seedset, direction, projection each key on `embedding_model_id` + memmap-sha; a rebuilt corpus silently invalidating a stale direction is the failure mode. **Enforcement:** refuse-on-mismatch at every consuming verb (jolt.py loader pattern); a mutated-sha round-trip test.
- **tvi#41 re-typed-id blast radius.** The tvi meta sidecar carries `NVEmbed:nvidia/NV-Embed-v2`. **Enforcement:** the canonical form is sourced from tvi's `corpus_join_manifest.json` (not the sidecar); a grep-gate forbids `NVEmbed:`/prefixed ids in artifact writers; the manifest flags the coordination. Not solved here (tvi#41's scope), not propagated.
- **Under-determined direction on thin (pattern×era) cells.** Small seed counts per era yield an unstable direction. **Enforcement:** bootstrap-cosine stability gate → `under-determined` verdict → refuse to promote; parallels `alignment_core`'s `min_pole_separation`.
- **Non-Gaussian projection distribution.** σ-thresholds mislead if the projection is skewed/multimodal. **Enforcement:** D3 records distribution shape; D5 switches to empirical-quantile thresholds when non-Gaussian.
- **Text sensitivity (security).** `top_exemplars`/`preview_pattern` read raw `chunk.text` (decades of personal corpus). **Boundary:** these are same-host in-process reads (sk-mcp on the operator's host); any *network* exposure inherits ADR-002's quote-abstraction (tvi-side) and is out of scope — no network door is added here.
- **Observability.** Every artifact's manifest (μ-sha, direction verdict + diagnostics, projection source shas, rate-table threshold+precision) is the health surface: a cold reader verifies the whole chain's integrity from manifests alone, no re-derivation. A build that can't reconcile μ_norm or a projection that can't match the memmap row count fails loud, never emits a silently-partial artifact.

## Open Questions

- [ ] **Shared projection kernel factoring.** Should `alignment_core`'s `build_axis`/`null_stats` be extracted into a `semantic_kinematics/projection/` module both `axis_alignment.py` and `direction.py` import (cleaner ONE-DOOR), or does `direction.py` import from `axis_alignment`? **Resolution:** decide at Phase 2 implementation; the ONE-DOOR test (identical projection given identical poles) is the acceptance regardless of which factoring wins.
- [ ] **Eigenbasis `k`.** Top-256 is a guess anchored on centered PR≈34 (≈8× headroom). **Resolution:** set k from the calibration run's cumulative-variance curve (`measure_cone.py` already reports it at k∈{1,5,10,34,50,100}); pick the k covering ~99% centered variance, recorded in the manifest.
- [ ] **`cosine` null treatment** (carried from ADR-SKMCP-0002 OQ). Is the direction-cosine reported raw or z-scored? **Resolution:** decide when D5's distribution-shape data exists; raw if interpretable at 4096-d after centering, else standardized.
- [ ] **Position-regime naming realignment.** ADR-SKMCP-0002 flagged renaming `analyze_axis_alignment` → `initialize_axis_analysis`/`run_axis_analysis`. This ADR's `initialize_direction`/`project_*` naming should converge with that if/when the rename lands. **Resolution:** out of scope here; a rename of shipped tools tracked with the 0002 open question.

## Preconditions

- **TVI-008 substrate must exist** — `corpus.db`, `vectors_nv-embed-v2.f32` + `corpus_join_manifest.json`, and at least one `<pattern_id>.seedset.json` (TVI-008 Phases 1–4). This ADR designs against the *frozen boundary contract*; the artifacts themselves are a dependency, not re-litigated here. Phase 1 (calibration) needs only the memmap+manifest; Phase 2+ need a seedset.
- **Canonical embedding-model id availability.** The canonical `nvidia/NV-Embed-v2` must be readable from tvi's `corpus_join_manifest.json` (TVI-008 Open Question: `embed_model` tag may be a pinned constant pending ADR-TVI-006 Phase 0 mint import). Until canonical, sk-mcp reads the manifest's declared id and flags provenance; it does not mint its own. Non-blocking for the design; noted so the id chain is honest.

## Supersession Relationships

**Supersedes:** — (additive; generalizes the axis source of the ADR-SKMCP-0001 primitive / ADR-SKMCP-0002 contract, replaces no tool).
**Superseded by:** TBD — if the position and bearing and functional-direction regimes are unified under one "measure behavioral axis" surface (the ADR-SKMCP-0002 Superseded-by hypothesis), this contract folds into it.

## Notes

Honesty marker: the projection math is not new (ADR-SKMCP-0001 already imported it from representation-engineering / word-embedding bipolar directions), and the build-validated-artifact + one-call-consume contract is not new (ADR-001 → ADR-SKMCP-0002). What this ADR adds is one axis *source* — corpus-derived, topic-matched, mean-centered centroid differences — mounted on the existing door, plus the μ-calibration artifact the runbook already named as future work, plus the era-scoped extraction and cross-projection matrix that make Aim 3 a falsifiable instrument. The value is the *generalization held to one door* and the *validation discipline encoded as refusals* (leakage alarm, topic-control, bootstrap gate, identity refusal), not any new mathematics.
