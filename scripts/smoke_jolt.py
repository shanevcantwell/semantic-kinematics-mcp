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
    vault chunks.jsonl (turns) joined to embed_checkpoint.jsonl (chunk_id ->
    768-dim embeddinggemma vector). Same space as A; NOT re-embedded. Fed to the
    analyze_embeddings seam.

Magnitude alone cannot tell the two registers apart -- that is the next, unbuilt
bearing stage. This gate only asks: does each specimen spike at all?

Usage:
    python scripts/smoke_jolt.py

The vault files are large (183MB / 1.39GB); both are STREAMED, never loaded
wholesale. They are read-only.
"""

import json
import os
import sys

import numpy as np

# Allow running as a plain script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer  # noqa: E402
from semantic_kinematics.mcp.state_manager import StateManager  # noqa: E402

# --- Config (the :8082 embeddinggemma path the vault was embedded with) ---
SPECIMEN_A_PATH = os.path.join(_REPO_ROOT, "data", "absurdism", "bypass_dialogue.txt")

# Vault corpus location is environment-driven so the smoke test runs under any
# user, not just the corpus author (issue #34). Override THOUGHT_VAULT_VECTORS_DIR
# to point at a different thought-vault-integration output/vectors checkout.
VAULT_DIR = os.environ.get(
    "THOUGHT_VAULT_VECTORS_DIR",
    "/srv/dev/shanevcantwell/thought-vault-integration/output/vectors",
)
CHUNKS_PATH = os.path.join(VAULT_DIR, "chunks.jsonl")
CHECKPOINT_PATH = os.path.join(VAULT_DIR, "embed_checkpoint.jsonl")
TARGET_CONVERSATION_ID = "e7c2fe94-7960-414d-8d59-92fdd2ee2303"

EMBED_BACKEND = "lmstudio"
EMBED_BASE_URL = "http://localhost:8082/v1"
EMBED_MODEL = "embeddinggemma-300M-F32"

SPIKE_THRESHOLD = 0.3


def _build_analyzer():
    """TrajectoryAnalyzer wired to the live :8082 embeddinggemma adapter."""
    manager = StateManager()
    manager.set_backend(EMBED_BACKEND, base_url=EMBED_BASE_URL, model_name=EMBED_MODEL)
    return TrajectoryAnalyzer(manager, acceleration_spike_threshold=SPIKE_THRESHOLD)


def _truncate(text, n=120):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "..."


def _report_metrics(metrics, label_kind):
    """Print the standard metric block; return a small dict for the verdict."""
    accels = metrics.accelerations
    print(f"  step count          : {len(metrics.sentences)}")
    print(f"  acceleration_profile: {[round(float(a), 4) for a in accels.tolist()]}")
    print(f"  max_acceleration    : {metrics.max_acceleration:.4f} "
          f"@ index {metrics.max_acceleration_index}")
    print(f"  mean_acceleration   : {metrics.mean_acceleration:.4f}")
    if len(accels) > 0:
        print(f"  median acceleration : {float(np.median(accels)):.4f}")
        print(f"  accel std           : {float(np.std(accels)):.4f}")
    print(f"  deadpan_score       : {metrics.deadpan_score:.4f}")
    print(f"  acceleration_spikes : {len(metrics.acceleration_spikes)}")
    for s in metrics.acceleration_spikes:
        print(f"      - idx {s.index:>3}  mag {s.magnitude:.4f}  "
              f"isolation {s.isolation_score:.4f}  pos {s.position_ratio:.3f}")

    # The acceleration index i corresponds to the velocity change between
    # velocity[i] and velocity[i+1]; the displacement landing there ends at
    # embedding step (i+2). Report the step the peak "lands on".
    peak_step = metrics.max_acceleration_index + 2
    peak_step = min(peak_step, len(metrics.sentences) - 1)
    peak_text = metrics.sentences[peak_step]
    print(f"  peak lands on {label_kind} #{peak_step}:")
    print(f'      "{_truncate(peak_text)}"')

    top_isolation = max(
        (s.isolation_score for s in metrics.acceleration_spikes), default=0.0
    )
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
    spiked = (
        m["n_spikes"] >= 1
        and m["peak_to_median"] >= 3.0
        and m["top_isolation"] > 0.0
    )
    print(f"\n  VERDICT [{name}]: ", end="")
    if spiked:
        print("ISOLATED SPIKE present.")
    else:
        print("FLAT / no isolated spike.")
    print(f"    rests on: peak_to_median={m['peak_to_median']:.2f}, "
          f"top_isolation={m['top_isolation']:.4f}, "
          f"deadpan={m['deadpan_score']:.4f}, n_spikes={m['n_spikes']}")
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

def _load_specimen_b_turns():
    """Stream chunks.jsonl -> ordered (chunk_id, text) for the target conv.

    Excludes speaker=='thinking'; keeps user+assistant ordered by message_index.
    """
    if not os.path.exists(CHUNKS_PATH):
        raise SystemExit(f"FAILURE: vault chunks not found at {CHUNKS_PATH}")
    turns = []
    needle = f'"{TARGET_CONVERSATION_ID}"'
    with open(CHUNKS_PATH, "r") as f:
        for line in f:
            # Cheap pre-filter before JSON parse (file is 183MB).
            if needle not in line:
                continue
            obj = json.loads(line)
            if obj.get("conversation_id") != TARGET_CONVERSATION_ID:
                continue
            if obj.get("speaker") == "thinking":
                continue
            if obj.get("speaker") not in ("user", "assistant"):
                continue
            turns.append((
                obj["chunk_id"],
                int(obj.get("message_index", 0)),
                obj.get("text", ""),
            ))
    turns.sort(key=lambda t: t[1])  # by message_index
    return [(cid, text) for (cid, _mi, text) in turns]


def _load_embeddings_for(chunk_ids):
    """Stream embed_checkpoint.jsonl -> {chunk_id: vector} for wanted ids only.

    The checkpoint is 1.39GB; we read line-by-line and keep only the ids in the
    wanted set, stopping early once all are found.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        raise SystemExit(f"FAILURE: vault checkpoint not found at {CHECKPOINT_PATH}")
    wanted = set(chunk_ids)
    found = {}
    with open(CHECKPOINT_PATH, "r") as f:
        for line in f:
            if not wanted:
                break
            # Cheap pre-filter: only parse lines that could contain a wanted id.
            obj = json.loads(line)
            cid = obj.get("chunk_id")
            if cid in wanted:
                emb = obj.get("embedding")
                if emb is None:
                    raise SystemExit(
                        f"FAILURE: checkpoint record for {cid} has no 'embedding'"
                    )
                found[cid] = np.asarray(emb, dtype=float)
                wanted.discard(cid)
    return found


def run_specimen_b(analyzer):
    print()
    print("=" * 72)
    print("SPECIMEN B -- escalation jolt (turn atom, precomputed vault vectors)")
    print("=" * 72)
    turns = _load_specimen_b_turns()
    if len(turns) < 2:
        raise SystemExit(
            f"FAILURE: only {len(turns)} usable turns for conversation "
            f"{TARGET_CONVERSATION_ID}; need >= 2"
        )
    chunk_ids = [cid for (cid, _t) in turns]
    emb_map = _load_embeddings_for(chunk_ids)

    missing = [cid for cid in chunk_ids if cid not in emb_map]
    if missing:
        raise SystemExit(
            f"FAILURE: {len(missing)} of {len(chunk_ids)} turn embeddings missing "
            f"from checkpoint (e.g. {missing[:3]}). Refusing to fabricate vectors."
        )

    # Stack in turn order; carry texts as labels.
    matrix = np.array([emb_map[cid] for cid in chunk_ids])
    labels = [text for (_cid, text) in turns]
    print(f"  joined {len(chunk_ids)} turns to precomputed {matrix.shape[1]}-dim vectors")

    metrics = analyzer.analyze_embeddings(matrix, labels=labels)
    return _report_metrics(metrics, "turn")


def main():
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
    m_b = run_specimen_b(analyzer)

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
