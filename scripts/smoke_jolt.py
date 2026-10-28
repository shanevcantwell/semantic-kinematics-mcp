#!/usr/bin/env python
"""
Stage-1 falsification smoke test for the register-disambiguation hypothesis.

Two text specimens, each producing a trajectory through embedding space, should
each show an ISOLATED ACCELERATION SPIKE against an otherwise flat baseline. If
either is flat, the shared-signature hypothesis is FALSIFIED at Stage 1 -- a
valid, important result, reported plainly, not massaged.

Specimen A -- comedic/absurdist jolt, SENTENCE atom:
    data/absurdism/bypass_dialogue.txt, run through the live sentence-wise path
    (analyze(text)) on the embeddinggemma server at :8082.

Specimen B -- conversational escalation jolt, TURN atom, PRECOMPUTED vectors:
    verified TVI canonical rows supply scoped turn IDs/text/metadata. A
    checkpoint may supply only valid chunk_id -> 768-dimensional vectors after
    its source/model sidecars are verified. The one historical sidecar-less
    artifact is accepted only by an exact digest + canonical-parity exception;
    its model identity remains non-authoritative, so it cannot participate in
    the default cross-specimen/shared-space verdict. Fed to analyze_embeddings.

Magnitude alone cannot tell the two registers apart -- that is the next, unbuilt
bearing stage. This gate only asks: does each specimen spike at all?

Usage:
    python scripts/smoke_jolt.py

The canonical query stream and vector checkpoint are read-only. Canonical
verified-open/query failures are fatal and never fall back to TVI JSONL.
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Allow running as a plain script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.embed_corpus import source_sidecar_path
from scripts.tvi_canonical_input import (
    DEFAULT_EXECUTABLE,
    CanonicalQueryProvenance,
    open_canonical_chunks,
)
from semantic_kinematics.embeddings.bulk import is_valid_embedding_record
from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer
from semantic_kinematics.mcp.state_manager import StateManager

# --- Config (specimen A's live :8082 embeddinggemma path) ---
SPECIMEN_A_PATH = os.path.join(_REPO_ROOT, "data", "absurdism", "bypass_dialogue.txt")

# TVI text/metadata comes only from the promoted canonical query door. The
# vectors directory remains independently configurable because its checkpoint
# is a derived vector boundary rather than corpus authority. Sidecars determine
# whether its model/source identity is authoritative.
DEFAULT_CORPUS_DB = os.environ.get(
    "THOUGHT_VAULT_CANONICAL_DB",
    "/srv/dev/shanevcantwell/thought-vault-integration/output/canonical/corpus.db",
)
DEFAULT_CANONICAL_ROOT = os.environ.get("THOUGHT_VAULT_CANONICAL_ROOT")
DEFAULT_TVI_EXECUTABLE = os.environ.get("TVI_CANONICAL_CORPUS_EXECUTABLE", DEFAULT_EXECUTABLE)
VECTORS_DIR = os.environ.get(
    "THOUGHT_VAULT_VECTORS_DIR",
    "/srv/dev/shanevcantwell/thought-vault-integration/output/vectors",
)
CHECKPOINT_PATH = os.path.join(VECTORS_DIR, "embed_checkpoint.jsonl")
TARGET_CONVERSATION_ID = "e7c2fe94-7960-414d-8d59-92fdd2ee2303"
SPECIMEN_B_INPUT_KEYS = ("base",)
SPECIMEN_B_SPEAKERS = ("user", "assistant")
SPECIMEN_B_ORDERING = "message_index_ascending_stable"

EMBED_BACKEND = "lmstudio"
EMBED_BASE_URL = "http://localhost:8082/v1"
EMBED_MODEL = "embeddinggemma-300M-F32"
CHECKPOINT_DIMENSIONS = 768

# Narrowly owned compatibility exception for the immutable historical specimen-B
# artifact. It exists only because that artifact predates .meta/.source sidecars.
# The source/query and selected text sequence are independently pinned, while
# model identity is explicitly *not* inferred from dimension or filename.
# Retirement trigger: replace this checkpoint with a sidecar-bound checkpoint
# produced by EMBED_MODEL, then delete these constants and the exception branch.
LEGACY_CHECKPOINT_SHA256 = "fb75d130e732ec6e7b3b919acd50b6600a2821920c8971ab01edc88cbaa35305"
LEGACY_CANONICAL_GENERATION_ID = (
    "tvi-generation-v1:sha256:5e3cae0cf5d8ecd5a91608301fed6d7bee5c5cd717d0cf8c23c8b886a0e30be3"
)
LEGACY_CANONICAL_QUERY_FINGERPRINT = (
    "b872ea1b85fa8ce91882fdc7377e5ebf058c25514c0907ab26dda84d1d420c9f"
)
LEGACY_SPECIMEN_SEQUENCE_SHA256 = "9ccfe2e83268b858cb27a7298187c43ca606492bf9fda93f26ab06d0950f5f07"

SPIKE_THRESHOLD = 0.3


@dataclass(frozen=True)
class CheckpointVerification:
    checkpoint_sha256: str
    source_basis: str
    model_basis: str
    model_authoritative: bool


def _build_analyzer():
    """TrajectoryAnalyzer wired to the live :8082 embeddinggemma adapter."""
    manager = StateManager()
    manager.set_backend(EMBED_BACKEND, base_url=EMBED_BASE_URL, model_name=EMBED_MODEL)
    return TrajectoryAnalyzer(manager, acceleration_spike_threshold=SPIKE_THRESHOLD)


def _build_checkpoint_analyzer():
    """Analyzer for precomputed vectors; no live embedding backend is touched."""
    return TrajectoryAnalyzer(StateManager(), acceleration_spike_threshold=SPIKE_THRESHOLD)


def _truncate(text, n=120):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "..."


def _report_metrics(metrics, label_kind):
    """Print the standard metric block; return a small dict for the verdict."""
    accels = metrics.accelerations
    print(f"  step count          : {len(metrics.sentences)}")
    print(f"  acceleration_profile: {[round(float(a), 4) for a in accels.tolist()]}")
    print(
        f"  max_acceleration    : {metrics.max_acceleration:.4f} "
        f"@ index {metrics.max_acceleration_index}"
    )
    print(f"  mean_acceleration   : {metrics.mean_acceleration:.4f}")
    if len(accels) > 0:
        print(f"  median acceleration : {float(np.median(accels)):.4f}")
        print(f"  accel std           : {float(np.std(accels)):.4f}")
    print(f"  deadpan_score       : {metrics.deadpan_score:.4f}")
    print(f"  acceleration_spikes : {len(metrics.acceleration_spikes)}")
    for s in metrics.acceleration_spikes:
        print(
            f"      - idx {s.index:>3}  mag {s.magnitude:.4f}  "
            f"isolation {s.isolation_score:.4f}  pos {s.position_ratio:.3f}"
        )

    # The acceleration index i corresponds to the velocity change between
    # velocity[i] and velocity[i+1]; the displacement landing there ends at
    # embedding step (i+2). Report the step the peak "lands on".
    peak_step = metrics.max_acceleration_index + 2
    peak_step = min(peak_step, len(metrics.sentences) - 1)
    peak_text = metrics.sentences[peak_step]
    print(f"  peak lands on {label_kind} #{peak_step}:")
    print(f'      "{_truncate(peak_text)}"')

    top_isolation = max((s.isolation_score for s in metrics.acceleration_spikes), default=0.0)
    median_accel = float(np.median(accels)) if len(accels) else 0.0
    # Peak-to-median ratio: how far the peak stands above the typical step.
    ratio = (metrics.max_acceleration / median_accel) if median_accel > 1e-9 else float("inf")
    return {
        "max_acceleration": metrics.max_acceleration,
        "median_accel": median_accel,
        "peak_to_median": ratio,
        "top_isolation": top_isolation,
        "deadpan_score": metrics.deadpan_score,
        "n_spikes": len(metrics.acceleration_spikes),
    }


def _verdict_for(name, m):
    """Per-specimen: isolated spike against flat baseline? State the numbers.

    Criteria (all numbers printed above):
      - peak stands clearly above the typical step (peak_to_median >= 3, i.e.
        the background is comparatively flat),
      - the peak spike is reasonably isolated (top isolation_score > 0),
      - the deadpan_score (isolated-spike-against-calm composite) is non-trivial.
    """
    spiked = m["n_spikes"] >= 1 and m["peak_to_median"] >= 3.0 and m["top_isolation"] > 0.0
    print(f"\n  VERDICT [{name}]: ", end="")
    if spiked:
        print("ISOLATED SPIKE present.")
    else:
        print("FLAT / no isolated spike.")
    print(
        f"    rests on: peak_to_median={m['peak_to_median']:.2f}, "
        f"top_isolation={m['top_isolation']:.4f}, "
        f"deadpan={m['deadpan_score']:.4f}, n_spikes={m['n_spikes']}"
    )
    return spiked


# --- Specimen A: sentence-wise live path ---


def run_specimen_a(analyzer):
    print("=" * 72)
    print("SPECIMEN A -- absurdist jolt (sentence atom, live :8082)")
    print("=" * 72)
    if not os.path.exists(SPECIMEN_A_PATH):
        raise SystemExit(f"FAILURE: specimen A not found at {SPECIMEN_A_PATH}")
    with open(SPECIMEN_A_PATH, "r") as f:
        text = f.read()
    metrics = analyzer.analyze(text)  # sentence-splits + embeds on :8082
    return _report_metrics(metrics, "sentence")


# --- Specimen B: precomputed vault vectors via analyze_embeddings ---


def _ordered_specimen_b_turns(rows):
    """Preserve the legacy specimen selection and stable message-index order."""
    turns = []
    for row in rows:
        if row.get("conversation_id") != TARGET_CONVERSATION_ID:
            continue
        if row.get("speaker") not in SPECIMEN_B_SPEAKERS:
            continue
        turns.append(
            (
                row["chunk_id"],
                int(row.get("message_index", 0)),
                row.get("text", ""),
            )
        )
    turns.sort(key=lambda turn: turn[1])
    return [(chunk_id, text) for chunk_id, _message_index, text in turns]


def _load_specimen_b_turns(
    *,
    corpus_db=DEFAULT_CORPUS_DB,
    canonical_root=DEFAULT_CANONICAL_ROOT,
    executable=DEFAULT_TVI_EXECUTABLE,
):
    """Read scoped canonical turns and return them with verified provenance.

    Scope is explicit and query-bound: the ``base`` acquisition leg, the target
    source conversation ID across acquisitions, and only user/assistant rows.
    The local stable sort exactly preserves the former ``message_index`` rule.
    """
    with open_canonical_chunks(
        corpus_db,
        canonical_root=canonical_root,
        input_keys=SPECIMEN_B_INPUT_KEYS,
        speakers=SPECIMEN_B_SPEAKERS,
        all_acquisitions_conversation_ids=(TARGET_CONVERSATION_ID,),
        executable=executable,
    ) as stream:
        provenance = stream.provenance
        turns = _ordered_specimen_b_turns(stream)
    return turns, provenance


def _sequence_sha256(turns):
    digest = hashlib.sha256()
    for chunk_id, text in turns:
        for field in (chunk_id, text):
            encoded = field.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FAILURE: {label} {path} is unreadable: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"FAILURE: {label} {path} must contain a JSON object")
    return value


def _validate_authoritative_sidecars(checkpoint_path: Path, provenance):
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    source_path = Path(source_sidecar_path(str(checkpoint_path)))
    meta_exists = meta_path.is_file()
    source_exists = source_path.is_file()
    if not meta_exists and not source_exists:
        return None
    if not meta_exists or not source_exists:
        raise SystemExit(
            "FAILURE: checkpoint provenance is incomplete; authoritative use requires both "
            f"{meta_path.name} and {source_path.name}"
        )

    metadata = _read_json_object(meta_path, "checkpoint metadata")
    if metadata.get("model_name") != EMBED_MODEL:
        raise SystemExit(
            "FAILURE: checkpoint model identity does not match specimen A: "
            f"expected {EMBED_MODEL!r}, got {metadata.get('model_name')!r}"
        )
    if metadata.get("dimensions") != CHECKPOINT_DIMENSIONS:
        raise SystemExit(
            "FAILURE: checkpoint dimensions do not match the smoke contract: "
            f"expected {CHECKPOINT_DIMENSIONS}, got {metadata.get('dimensions')!r}"
        )

    source = _read_json_object(source_path, "checkpoint source provenance")
    query = source.get("query")
    production_fingerprint = (
        query.get("query_fingerprint_sha256") if isinstance(query, dict) else None
    )
    expected_filters = {
        "channels": [],
        "speakers": [],
        "conversations": [],
        "nonblank_text": None,
        "time_roles": [],
    }
    if (
        source.get("source_kind") != "tvi_canonical_db"
        or source.get("generation_id") != provenance.generation_id
        or source.get("database_sha256") != provenance.database_sha256
        or source.get("logical_digest_sha256") != provenance.logical_digest_sha256
        or source.get("normalized_input_keys") != list(SPECIMEN_B_INPUT_KEYS)
        or source.get("normalized_filters") != expected_filters
        or not isinstance(query, dict)
        or query.get("verb") != provenance.query_verb
        or query.get("ordering") != provenance.ordering
        or not isinstance(production_fingerprint, str)
        or len(production_fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in production_fingerprint)
    ):
        raise SystemExit(
            "FAILURE: checkpoint source sidecar does not bind the queried canonical "
            "generation/base acquisition"
        )
    return CheckpointVerification(
        checkpoint_sha256="",
        source_basis=(
            "authoritative .source.json: same canonical generation/logical digest, input_keys=base"
        ),
        model_basis=f"authoritative .meta.json: {EMBED_MODEL}/{CHECKPOINT_DIMENSIONS}",
        model_authoritative=True,
    )


def _validate_legacy_exception(checkpoint_sha256, provenance, turns):
    observed = {
        "checkpoint_sha256": checkpoint_sha256,
        "generation_id": provenance.generation_id,
        "query_fingerprint": provenance.query_fingerprint_sha256,
        "sequence_sha256": _sequence_sha256(turns),
    }
    expected = {
        "checkpoint_sha256": LEGACY_CHECKPOINT_SHA256,
        "generation_id": LEGACY_CANONICAL_GENERATION_ID,
        "query_fingerprint": LEGACY_CANONICAL_QUERY_FINGERPRINT,
        "sequence_sha256": LEGACY_SPECIMEN_SEQUENCE_SHA256,
    }
    if observed != expected:
        differences = ", ".join(
            f"{key}=observed:{observed[key]!r}/expected:{expected[key]!r}"
            for key in expected
            if observed[key] != expected[key]
        )
        raise SystemExit(
            "FAILURE: sidecar-less checkpoint is outside the pinned legacy exception: "
            + differences
        )
    return CheckpointVerification(
        checkpoint_sha256=checkpoint_sha256,
        source_basis=(
            "legacy exception: checkpoint SHA-256 + canonical generation/scoped-query "
            "+ exact turn text/ID sequence parity pinned"
        ),
        model_basis=(
            "legacy exception only: 768-dimensional vectors; producing model identity "
            "is not authoritative"
        ),
        model_authoritative=False,
    )


def _load_embeddings_for(chunk_ids, *, turns, provenance):
    """Load latest valid selected vectors and verify checkpoint provenance.

    The complete checkpoint is streamed so append-only retries are resolved and
    the artifact digest is verified. Failed, malformed, non-finite, wrong-width,
    and zero-norm records never become specimen vectors.
    """
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.is_file():
        raise SystemExit(f"FAILURE: vault checkpoint not found at {CHECKPOINT_PATH}")

    authoritative = _validate_authoritative_sidecars(checkpoint_path, provenance)
    wanted = set(chunk_ids)
    found = {}
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as checkpoint_file:
        for raw_line in checkpoint_file:
            digest.update(raw_line)
            try:
                obj = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict):
                continue
            cid = obj.get("chunk_id")
            if cid in wanted and is_valid_embedding_record(obj, CHECKPOINT_DIMENSIONS):
                # Latest valid retry wins. A failed/invalid retry never erases a
                # previously valid vector, matching the checkpoint status contract.
                found[cid] = np.asarray(obj["embedding"], dtype=float)

    checkpoint_sha256 = digest.hexdigest()
    if authoritative is None:
        verification = _validate_legacy_exception(checkpoint_sha256, provenance, turns)
    else:
        verification = CheckpointVerification(
            checkpoint_sha256=checkpoint_sha256,
            source_basis=authoritative.source_basis,
            model_basis=authoritative.model_basis,
            model_authoritative=True,
        )

    missing = [cid for cid in chunk_ids if cid not in found]
    if missing:
        raise SystemExit(
            f"FAILURE: {len(missing)} of {len(chunk_ids)} turn embeddings lack a valid "
            f"{CHECKPOINT_DIMENSIONS}-dimensional vector (e.g. {missing[:3]})."
        )
    return found, verification


def _require_authoritative_shared_model(verification: CheckpointVerification):
    if not verification.model_authoritative:
        raise SystemExit(
            "FAILURE: refusing the cross-specimen/shared-space verdict because specimen B's "
            "historical checkpoint has no authoritative model identity. Run "
            "--specimen-b-only for the pinned legacy specimen, or publish matching "
            ".meta.json/.source.json sidecars for an authoritative checkpoint."
        )


def _report_specimen_b_source(provenance: CanonicalQueryProvenance):
    print(
        "  canonical scope      : "
        f"input_keys={','.join(SPECIMEN_B_INPUT_KEYS)} "
        f"conversation_scope=all_acquisitions "
        f"conversation_id={TARGET_CONVERSATION_ID} "
        f"speakers={','.join(SPECIMEN_B_SPEAKERS)} "
        f"consumer_order={SPECIMEN_B_ORDERING}"
    )
    print(f"  canonical generation : {provenance.generation_id}")
    print(f"  logical digest       : {provenance.logical_digest_sha256}")
    print(
        "  canonical query      : "
        f"{provenance.query_verb}/{provenance.ordering} "
        f"fingerprint={provenance.query_fingerprint_sha256}"
    )


def run_specimen_b(analyzer, *, require_authoritative_shared_model=False):
    print()
    print("=" * 72)
    print("SPECIMEN B -- escalation jolt (turn atom, precomputed vault vectors)")
    print("=" * 72)
    turns, provenance = _load_specimen_b_turns()
    _report_specimen_b_source(provenance)
    if len(turns) < 2:
        raise SystemExit(
            f"FAILURE: only {len(turns)} usable turns for conversation "
            f"{TARGET_CONVERSATION_ID}; need >= 2"
        )
    chunk_ids = [cid for (cid, _t) in turns]
    emb_map, verification = _load_embeddings_for(
        chunk_ids,
        turns=turns,
        provenance=provenance,
    )
    print(f"  checkpoint SHA-256  : {verification.checkpoint_sha256}")
    print(f"  checkpoint source   : {verification.source_basis}")
    print(f"  checkpoint model    : {verification.model_basis}")
    if require_authoritative_shared_model:
        _require_authoritative_shared_model(verification)

    # Stack in turn order; carry texts as labels.
    matrix = np.array([emb_map[cid] for cid in chunk_ids])
    labels = [text for (_cid, text) in turns]
    print(f"  joined {len(chunk_ids)} turns to precomputed {matrix.shape[1]}-dim vectors")

    metrics = analyzer.analyze_embeddings(matrix, labels=labels)
    return _report_metrics(metrics, "turn")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specimen-b-only",
        action="store_true",
        help=(
            "run only the canonical-text + checkpoint-vector specimen; "
            "does not contact the live embedding endpoint"
        ),
    )
    args = parser.parse_args(argv)

    if args.specimen_b_only:
        metrics = run_specimen_b(_build_checkpoint_analyzer())
        print()
        print("=" * 72)
        print("CANONICAL CHECKPOINT SMOKE VERDICT")
        print("=" * 72)
        _verdict_for("Specimen B (escalation, turn)", metrics)
        return

    analyzer = _build_analyzer()

    # Fail fast & loud if the live embedder is unreachable -- specimen A needs it.
    try:
        analyzer.embed_sentences(["connectivity probe"])
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"FAILURE: cannot reach embeddinggemma at {EMBED_BASE_URL} "
            f"({type(e).__name__}: {e}). Not fabricating embeddings."
        )

    m_a = run_specimen_a(analyzer)
    m_b = run_specimen_b(analyzer, require_authoritative_shared_model=True)

    print()
    print("=" * 72)
    print("STAGE-1 VERDICT")
    print("=" * 72)
    a_spiked = _verdict_for("Specimen A (absurdist, sentence)", m_a)
    b_spiked = _verdict_for("Specimen B (escalation, turn)", m_b)

    print()
    if a_spiked and b_spiked:
        print("COMBINED: BOTH specimens show an isolated acceleration spike.")
        print("  -> shared-signature hypothesis SURVIVES this Stage-1 gate.")
        print("  NOTE: magnitude alone cannot yet distinguish the two registers;")
        print("  that is the next, unbuilt BEARING stage.")
    else:
        print("COMBINED: at least one specimen is FLAT.")
        print("  -> shared-signature hypothesis is FALSIFIED at Stage 1.")
        flat = []
        if not a_spiked:
            flat.append("A (absurdist/sentence)")
        if not b_spiked:
            flat.append("B (escalation/turn)")
        print(f"  flat specimen(s): {', '.join(flat)} (profiles shown above).")


if __name__ == "__main__":
    main()
