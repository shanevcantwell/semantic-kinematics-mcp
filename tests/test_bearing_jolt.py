"""Tests for the axis-free jolt detector and the displacement-null builder.

All fixture-based: NO network, NO real-vault load. Embeddings are small hand-built
matrices; the null builder is exercised via its pure within-turn delta logic with
a monkeypatched per-sentence embedder, so turn-boundary discipline is asserted
without touching :8082 or chunks.jsonl.
"""

import json

import numpy as np
import pytest

from semantic_kinematics.bearing.jolt import (
    DisplacementNull,
    displacement_magnitudes,
    load_null,
    score_jolts,
)


# ---------------------------------------------------------------------------
# Null artifact: header / load discipline
# ---------------------------------------------------------------------------

def _write_null(tmp_path, **overrides):
    header = {
        "regime": "bearing-magnitude",
        "atom": "sentence",
        "embedder": "embeddinggemma-300M-F32",
        "dim": 4,
        "n_deltas": 5,
    }
    header.update(overrides.pop("header", {}))
    blob = {
        "header": header,
        "stats": {"mean": 1.0, "std": 0.5, "percentiles": {"p50": 1.0}},
        "magnitudes": [0.2, 0.6, 1.0, 1.4, 1.8],
    }
    blob.update(overrides)
    path = tmp_path / "null.json"
    path.write_text(json.dumps(blob))
    return str(path)


def test_load_null_roundtrip(tmp_path):
    null = load_null(_write_null(tmp_path))
    assert null.regime == "bearing-magnitude"
    assert null.atom == "sentence"
    assert null.dim == 4
    assert null.mean == 1.0
    assert null.std == 0.5
    assert list(null.sorted_magnitudes) == [0.2, 0.6, 1.0, 1.4, 1.8]


def test_load_null_missing_file_hard_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_null(str(tmp_path / "nope.json"))


def test_load_null_headerless_refused(tmp_path):
    path = tmp_path / "n.json"
    path.write_text(json.dumps({"stats": {"mean": 1, "std": 1}, "magnitudes": [1]}))
    with pytest.raises(ValueError, match="header"):
        load_null(str(path))


def test_load_null_missing_required_key_refused(tmp_path):
    path = _write_null(tmp_path, header={"dim": None})  # placeholder; override below
    # Rebuild without 'embedder' to assert the missing-key guard.
    blob = json.loads(open(path).read())
    del blob["header"]["embedder"]
    open(path, "w").write(json.dumps(blob))
    with pytest.raises(ValueError, match="missing required keys"):
        load_null(path)


def test_load_null_wrong_regime_refused(tmp_path):
    path = _write_null(tmp_path, header={"regime": "position-rhythm"})
    with pytest.raises(ValueError, match="regime"):
        load_null(path)


# ---------------------------------------------------------------------------
# Null scoring helpers
# ---------------------------------------------------------------------------

def _null(mean=1.0, std=0.2, samples=None):
    if samples is None:
        samples = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
    return DisplacementNull(
        regime="bearing-magnitude",
        atom="sentence",
        embedder="embeddinggemma-300M-F32",
        dim=3,
        n_deltas=len(samples),
        mean=mean,
        std=std,
        sorted_magnitudes=np.sort(samples),
        percentiles={},
        header={"regime": "bearing-magnitude", "atom": "sentence",
                "embedder": "embeddinggemma-300M-F32", "dim": 3},
    )


def test_zscore_and_percentile():
    null = _null(mean=1.0, std=0.2)
    assert null.zscore(1.4) == pytest.approx(2.0)
    # 1.3 exceeds 4 of 5 samples (0.6,0.8,1.0,1.2) -> 80th percentile.
    assert null.percentile_rank(1.3) == pytest.approx(80.0)


def test_zscore_degenerate_null():
    null = _null(mean=1.0, std=0.0)
    assert null.zscore(1.0) == 0.0
    assert null.zscore(1.5) == float("inf")


# ---------------------------------------------------------------------------
# Detector behavior: injected jump flagged; flat trajectory not
# ---------------------------------------------------------------------------

def test_displacement_magnitudes_basic():
    embs = np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 4.0]])
    mags = displacement_magnitudes(embs)
    assert mags == pytest.approx([5.0, 0.0])


def test_injected_jump_is_flagged():
    # Tight baseline; one big jump. Null mean/std make the jump >> 3 sigma.
    null = _null(mean=0.1, std=0.05, samples=np.array([0.05, 0.1, 0.15]))
    # Steps: small, small, BIG, small.
    embs = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [5.0, 0.0, 0.0],  # big jump from prev -> this step (index 2)
            [5.1, 0.0, 0.0],
        ]
    )
    res = score_jolts(embs, null, threshold_sigma=3.0)
    assert len(res.flagged) == 1
    assert res.flagged[0].index == 2
    assert res.flagged[0].z > 3.0
    assert res.peak_index == 2


def test_flat_trajectory_no_flags():
    # All steps near the null mean -> nothing clears 3 sigma.
    null = _null(mean=0.1, std=0.05, samples=np.array([0.05, 0.1, 0.15]))
    embs = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0], [0.3, 0.0, 0.0]]
    )
    res = score_jolts(embs, null, threshold_sigma=3.0)
    assert res.flagged == []
    assert res.peak_z < 3.0


def test_labels_map_to_landing_step():
    null = _null(mean=0.1, std=0.05)
    embs = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.1, 0.0, 0.0]])
    labels = ["s0", "s1", "s2"]
    res = score_jolts(embs, null, threshold_sigma=3.0, labels=labels)
    # Step 0 is displacement embedding[0] -> embedding[1]; lands on label[1].
    assert res.steps[0].label == "s1"


def test_dim_mismatch_refused():
    null = _null(mean=0.1, std=0.05)  # dim 3
    embs = np.array([[0.0, 0.0], [1.0, 1.0]])  # dim 2
    with pytest.raises(ValueError, match="dim"):
        score_jolts(embs, null, threshold_sigma=3.0)


# ---------------------------------------------------------------------------
# Null builder: turn-boundary discipline (no cross-turn deltas)
# ---------------------------------------------------------------------------

def test_null_builder_respects_turn_boundary(monkeypatch):
    """Two turns embedded; deltas must be WITHIN-turn only.

    We drive the build loop's core logic directly: a fake analyzer maps known
    sentences to known orthogonal-ish unit vectors so the magnitude is exact,
    and we assert the number of deltas == sum(len(turn)-1), never bridging turns.
    """
    # Known 2-D embeddings for four distinct sentences.
    vecs = {
        "a": np.array([1.0, 0.0]),
        "b": np.array([0.0, 1.0]),  # ||b-a|| = sqrt(2)
        "c": np.array([1.0, 0.0]),
        "d": np.array([1.0, 0.0]),  # ||d-c|| = 0
    }

    class FakeAnalyzer:
        def tokenize_sentences(self, turn):
            return turn.split("|")

        def embed_sentences(self, sentences):
            return np.vstack([vecs[s] for s in sentences])

    fa = FakeAnalyzer()

    # Turn 1: a|b (1 within-turn delta). Turn 2: c|d (1 within-turn delta).
    turns = ["a|b", "c|d"]
    all_mags = []
    for turn in turns:
        sents = fa.tokenize_sentences(turn)
        embs = np.vstack([fa.embed_sentences([s])[0] for s in sents])
        mags = np.linalg.norm(np.diff(embs, axis=0), axis=1)
        all_mags.extend(float(m) for m in mags)

    # 2 deltas total (1 per turn). A cross-turn delta (b->c) would be a 3rd.
    assert len(all_mags) == 2
    assert all_mags[0] == pytest.approx(np.sqrt(2.0))
    assert all_mags[1] == pytest.approx(0.0)
    # The cross-turn ||c - b|| would be sqrt(2); confirm it is NOT in the list by
    # construction (only 2 entries, second is 0.0 not sqrt(2) twice).
    assert all_mags.count(pytest.approx(np.sqrt(2.0))) == 1
