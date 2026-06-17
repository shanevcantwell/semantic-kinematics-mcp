"""Tests for the context-conditioned phrase-displacement null (ADR-SKMCP-0003).

All fixture-based: NO network, NO real-vault load, NO embedder. The conditioned
null is built by hand from known stratum statistics so loader hard-fails,
stratum selection, sparsity backoff (with the recorded level), and the
z-score/percentile math are asserted without touching :8083 or chunks.jsonl.

Mirrors tests/test_bearing_jolt.py style.
"""

import json

import numpy as np
import pytest

from semantic_kinematics.bearing.jolt import (
    ConditionedDisplacementNull,
    ConditionedStratum,
    load_conditioned_null,
)


# ---------------------------------------------------------------------------
# Conditioned null artifact: header / load discipline
# ---------------------------------------------------------------------------

def _write_conditioned_null(tmp_path, **overrides):
    header = {
        "regime": "bearing-magnitude-conditioned",
        "atom": "phrase-conditioned",
        "embedder": "embeddinggemma-300M-F32-nonpooled",
        "dim": 768,
        "k_range": [0, 1, 2, 3, 4, 5],
    }
    header.update(overrides.pop("header", {}))
    blob = {
        "header": header,
        "strata": {
            "k2|4-7|SET_QUOTE": {
                "mean": 1.0,
                "std": 0.5,
                "n": 250,
                "percentiles": {"p50": 1.0},
                "sorted_magnitudes": [0.2, 0.6, 1.0, 1.4, 1.8],
            }
        },
    }
    blob.update(overrides)
    path = tmp_path / "cond_null.json"
    path.write_text(json.dumps(blob))
    return str(path)


def test_load_conditioned_null_roundtrip(tmp_path):
    null = load_conditioned_null(_write_conditioned_null(tmp_path))
    assert null.regime == "bearing-magnitude-conditioned"
    assert null.atom == "phrase-conditioned"
    assert null.dim == 768
    assert null.k_range == [0, 1, 2, 3, 4, 5]
    cell = null.strata["k2|4-7|SET_QUOTE"]
    assert cell.mean == 1.0
    assert cell.std == 0.5
    assert cell.n == 250
    assert list(cell.sorted_magnitudes) == [0.2, 0.6, 1.0, 1.4, 1.8]


def test_load_conditioned_null_missing_file_hard_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_conditioned_null(str(tmp_path / "nope.json"))


def test_load_conditioned_null_headerless_refused(tmp_path):
    path = tmp_path / "n.json"
    path.write_text(json.dumps({"strata": {"k0": {"mean": 1, "std": 1, "n": 1,
                                                  "sorted_magnitudes": [1]}}}))
    with pytest.raises(ValueError, match="header"):
        load_conditioned_null(str(path))


def test_load_conditioned_null_missing_required_key_refused(tmp_path):
    path = _write_conditioned_null(tmp_path)
    blob = json.loads(open(path).read())
    del blob["header"]["k_range"]
    open(path, "w").write(json.dumps(blob))
    with pytest.raises(ValueError, match="missing required keys"):
        load_conditioned_null(path)


def test_load_conditioned_null_wrong_regime_refused(tmp_path):
    path = _write_conditioned_null(tmp_path, header={"regime": "bearing-magnitude"})
    with pytest.raises(ValueError, match="regime"):
        load_conditioned_null(path)


def test_load_conditioned_null_missing_strata_refused(tmp_path):
    path = _write_conditioned_null(tmp_path, strata={})
    with pytest.raises(ValueError, match="strata"):
        load_conditioned_null(path)


# ---------------------------------------------------------------------------
# Stratum scoring helpers (z-score / percentile correctness)
# ---------------------------------------------------------------------------

def test_stratum_zscore_and_percentile():
    cell = ConditionedStratum(
        mean=1.0,
        std=0.2,
        n=5,
        sorted_magnitudes=np.array([0.6, 0.8, 1.0, 1.2, 1.4]),
        percentiles={},
    )
    assert cell.zscore(1.4) == pytest.approx(2.0)
    # 1.3 exceeds 4 of 5 samples -> 80th percentile.
    assert cell.percentile_rank(1.3) == pytest.approx(80.0)


def test_stratum_zscore_degenerate():
    cell = ConditionedStratum(mean=1.0, std=0.0, n=3,
                              sorted_magnitudes=np.array([1.0, 1.0, 1.0]),
                              percentiles={})
    assert cell.zscore(1.0) == 0.0
    assert cell.zscore(1.5) == float("inf")


# ---------------------------------------------------------------------------
# Stratum selection + sparsity backoff (records the level used)
# ---------------------------------------------------------------------------

def _cell(mean, std, n, samples):
    return ConditionedStratum(
        mean=mean, std=std, n=n,
        sorted_magnitudes=np.sort(np.asarray(samples, dtype=float)),
        percentiles={},
    )


def _null(strata):
    return ConditionedDisplacementNull(
        regime="bearing-magnitude-conditioned",
        atom="phrase-conditioned",
        embedder="embeddinggemma-300M-F32-nonpooled",
        dim=768,
        k_range=[0, 1, 2, 3, 4, 5],
        strata=strata,
        header={"regime": "bearing-magnitude-conditioned"},
    )


def test_selection_picks_finest_cell_when_populated():
    null = _null({
        "k2|4-7|SET_QUOTE": _cell(1.0, 0.2, 300, [0.6, 0.8, 1.0, 1.2, 1.4]),
        "k2|4-7":           _cell(5.0, 5.0, 300, [5.0]),
        "k2":               _cell(9.0, 9.0, 300, [9.0]),
    })
    score = null.score_step(1.4, k=2, length_bucket="4-7",
                            demarcator="SET_QUOTE", n_min=200)
    assert score.backoff_level == "k|length|demarcator"
    assert score.stratum_key == "k2|4-7|SET_QUOTE"
    assert score.n == 300
    assert score.z == pytest.approx(2.0)  # against the finest cell's mean/std


def test_backoff_to_k_length_when_finest_below_nmin():
    null = _null({
        "k2|4-7|SET_QUOTE": _cell(99.0, 1.0, 50, [99.0]),   # below n_min
        "k2|4-7":           _cell(1.0, 0.2, 300, [0.6, 0.8, 1.0, 1.2, 1.4]),
        "k2":               _cell(9.0, 9.0, 300, [9.0]),
    })
    score = null.score_step(1.4, k=2, length_bucket="4-7",
                            demarcator="SET_QUOTE", n_min=200)
    assert score.backoff_level == "k|length"
    assert score.stratum_key == "k2|4-7"
    assert score.n == 300
    assert score.z == pytest.approx(2.0)  # against the (k,length) cell


def test_backoff_to_k_when_finer_cells_below_nmin():
    null = _null({
        "k2|4-7|SET_QUOTE": _cell(99.0, 1.0, 10, [99.0]),   # below n_min
        "k2|4-7":           _cell(88.0, 1.0, 30, [88.0]),   # below n_min
        "k2":               _cell(1.0, 0.2, 500, [0.6, 0.8, 1.0, 1.2, 1.4]),
    })
    score = null.score_step(1.4, k=2, length_bucket="4-7",
                            demarcator="SET_QUOTE", n_min=200)
    assert score.backoff_level == "k"
    assert score.stratum_key == "k2"
    assert score.n == 500
    assert score.z == pytest.approx(2.0)  # against the k cell


def test_backoff_skips_missing_finer_cells():
    # Finer cells simply absent (not just thin) -> still falls through to k.
    null = _null({
        "k3": _cell(1.0, 0.2, 400, [0.6, 0.8, 1.0, 1.2, 1.4]),
    })
    score = null.score_step(1.2, k=3, length_bucket="8-15",
                            demarcator="TERM_FLOW", n_min=200)
    assert score.backoff_level == "k"
    assert score.stratum_key == "k3"


def test_all_cells_below_nmin_hard_fails():
    null = _null({
        "k2|4-7|SET_QUOTE": _cell(1.0, 1.0, 10, [1.0]),
        "k2|4-7":           _cell(1.0, 1.0, 20, [1.0]),
        "k2":               _cell(1.0, 1.0, 30, [1.0]),
    })
    with pytest.raises(ValueError, match="too thin to calibrate"):
        null.score_step(1.0, k=2, length_bucket="4-7",
                        demarcator="SET_QUOTE", n_min=200)


def test_no_matching_k_cell_hard_fails():
    null = _null({"k0": _cell(1.0, 1.0, 500, [1.0])})
    with pytest.raises(ValueError, match="under-populated null"):
        null.score_step(1.0, k=2, length_bucket="4-7",
                        demarcator="SET_QUOTE", n_min=200)
