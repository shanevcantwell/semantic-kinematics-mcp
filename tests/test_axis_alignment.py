"""
Tests for the referential axis-alignment analysis.

The numerics are exercised through pure functions with hand-built embeddings on
a known axis -- no spaCy, no real embedding backend. The async handler is
exercised with a deterministic lookup adapter and a monkeypatched tokenizer, so
the whole suite is CI-safe.
"""

import asyncio

import numpy as np
import pytest

from semantic_kinematics.mcp.commands import axis_alignment as ax
from semantic_kinematics.mcp.commands.axis_alignment import (
    alignment_core,
    build_axis,
    build_null_cache,
    load_null_cache,
    analyze_axis_alignment,
)
from semantic_kinematics.mcp.state_manager import StateManager


# Axis along the first basis vector of a small space; a symmetric null centered
# at the origin -> null mean projects to 0, with nonzero spread along the axis.
NULL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0, 0.0],
])
POS = np.array([[1.0, 0.0, 0.0, 0.0]])      # positive pole on +axis
NEG = np.array([[-1.0, 0.0, 0.0, 0.0]])     # negative pole on -axis


class FakeAdapter:
    """Deterministic lookup adapter: exact text -> chosen vector."""

    def __init__(self, vectors, model_name="fake-1", dimensions=4):
        self._vectors = vectors
        self._model_name = model_name
        self._dimensions = dimensions

    @property
    def model_name(self):
        return self._model_name

    @property
    def dimensions(self):
        return self._dimensions

    def embed(self, text):
        return np.asarray(self._vectors[text], dtype=float)

    def embed_batch(self, texts):
        return np.array([self.embed(t) for t in texts])


# --------------------------------------------------------------------------- #
# Pure numerics
# --------------------------------------------------------------------------- #

def test_build_axis_normalizes_and_reports_separation():
    unit, sep = build_axis(POS, NEG[0])
    np.testing.assert_allclose(unit, [1.0, 0.0, 0.0, 0.0])
    assert sep == pytest.approx(2.0)


def test_aligned_march_gives_positive_drift_and_high_straightness():
    # Sentences marching monotonically along +axis (proj 0 -> 0.5 -> 1.0).
    sentences = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    res = alignment_core(sentences, POS, NEG, NULL)
    assert "error" not in res
    assert res["axis_drift"] > 0
    assert res["axis_straightness"] == pytest.approx(1.0)  # net == total step
    # z trace strictly increasing
    zs = res["position_zscores"]
    assert zs[0] < zs[1] < zs[2]


def test_orthogonal_march_is_flat_on_the_axis():
    # Movement only along axis-orthogonal directions -> constant projection.
    sentences = np.array([
        [0.3, 1.0, 0.0, 0.0],
        [0.3, 0.0, 1.0, 0.0],
        [0.3, 0.0, 0.0, 1.0],
    ])
    res = alignment_core(sentences, POS, NEG, NULL)
    assert res["axis_drift"] == pytest.approx(0.0, abs=1e-9)
    assert res["axis_straightness"] == pytest.approx(0.0)  # zero net, zero total


def test_omitted_negative_pole_uses_null_mean():
    sentences = np.array([
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ])
    # NULL mean is the origin, so axis = pos_mean - 0 = +axis (separation 1.0).
    res = alignment_core(sentences, POS, None, NULL)
    assert "error" not in res
    assert res["pole_separation"] == pytest.approx(1.0)
    assert res["axis_drift"] > 0


def test_colliding_anchors_gate_fires():
    sentences = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    near = np.array([[1.0, 0.0, 0.0, 0.0]])
    res = alignment_core(sentences, near, near, NULL, min_pole_separation=0.05)
    assert res["error"] == "axis underdetermined"
    assert res["pole_separation"] == pytest.approx(0.0)


def test_degenerate_null_variance_errors():
    sentences = np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
    flat_null = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])  # no spread on axis
    res = alignment_core(sentences, POS, NEG, flat_null)
    assert "zero variance" in res["error"]


def test_too_few_sentences_errors():
    res = alignment_core(np.array([[1.0, 0.0, 0.0, 0.0]]), POS, NEG, NULL)
    assert "at least 2 sentences" in res["error"]


# --------------------------------------------------------------------------- #
# Null-cache IO
# --------------------------------------------------------------------------- #

def test_null_cache_round_trip(tmp_path):
    adapter = FakeAdapter({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    out = str(tmp_path / "null.npy")
    manifest = build_null_cache(adapter, ["a", "b"], out, source="unit-test")
    assert manifest["model_name"] == "fake-1"
    assert manifest["count"] == 2 and manifest["dimensions"] == 4

    emb, loaded = load_null_cache(out + ".json")
    assert emb.shape == (2, 4)
    assert loaded["model_name"] == "fake-1"
    assert loaded["source"] == "unit-test"


def test_build_null_cache_normalizes_extension(tmp_path):
    adapter = FakeAdapter({"a": [1, 0, 0, 0], "b": [0, 1, 0, 0]})
    out = str(tmp_path / "null")  # no .npy
    build_null_cache(adapter, ["a", "b"], out)
    # manifest + npy resolve consistently
    emb, _ = load_null_cache(out + ".npy.json")
    assert emb.shape == (2, 4)


# --------------------------------------------------------------------------- #
# Async handler plumbing
# --------------------------------------------------------------------------- #

def _make_null(tmp_path, adapter):
    out = str(tmp_path / "null.npy")
    build_null_cache(adapter, ["N0", "N1", "N2", "N3"], out)
    return out + ".json"


@pytest.fixture
def vectors():
    return {
        "N0": [1, 0, 0, 0], "N1": [-1, 0, 0, 0],
        "N2": [0, 1, 0, 0], "N3": [0, -1, 0, 0],
        "POS": [1, 0, 0, 0],
        "S0": [0, 1, 0, 0], "S1": [0.5, 0.5, 0, 0], "S2": [1.0, 0, 0, 0],
    }


def test_handler_happy_path(tmp_path, monkeypatch, vectors):
    adapter = FakeAdapter(vectors)
    manager = StateManager()
    manager._adapter = adapter
    manifest = _make_null(tmp_path, adapter)

    from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer
    monkeypatch.setattr(TrajectoryAnalyzer, "tokenize_sentences",
                        lambda self, text: ["S0", "S1", "S2"])

    res = asyncio.run(analyze_axis_alignment(manager, {
        "text": "ignored, tokenizer is patched",
        "anchor_positive": "POS",
        "background_ref": manifest,
    }))
    assert "error" not in res
    assert res["n_sentences"] == 3
    assert res["axis_drift"] > 0
    assert res["axis_straightness"] == pytest.approx(1.0)
    assert res["model_name"] == "fake-1"


def test_handler_requires_background(monkeypatch, vectors):
    monkeypatch.delenv("AXIS_NULL_MANIFEST", raising=False)
    manager = StateManager()
    manager._adapter = FakeAdapter(vectors)
    res = asyncio.run(analyze_axis_alignment(manager, {
        "text": "x", "anchor_positive": "POS",
    }))
    assert "background_ref" in res["error"]


def test_handler_rejects_model_mismatch(tmp_path, vectors):
    build_adapter = FakeAdapter(vectors, model_name="other-model")
    manifest = _make_null(tmp_path, build_adapter)

    manager = StateManager()
    manager._adapter = FakeAdapter(vectors, model_name="fake-1")
    res = asyncio.run(analyze_axis_alignment(manager, {
        "text": "x", "anchor_positive": "POS", "background_ref": manifest,
    }))
    assert "geometry differs" in res["error"]


def test_handler_empty_anchor_errors(tmp_path, vectors):
    manager = StateManager()
    manager._adapter = FakeAdapter(vectors)
    res = asyncio.run(analyze_axis_alignment(manager, {
        "text": "x", "anchor_positive": "   ", "background_ref": "ignored",
    }))
    assert "anchor_positive" in res["error"]
