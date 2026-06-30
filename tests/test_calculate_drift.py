"""Regression tests for the calculate_drift MCP command handler (issue #35).

Root cause guarded here: calculate_drift used to `import` a non-existent module
`semantic_kinematics.prompt_geometry.metrics` for a free `cosine_distance`
function. That module never existed in repo history; the canonical
`cosine_distance` is a method on the `EmbeddingAdapter` (ONE-MINT), reached
through the `StateManager.get_adapter()` one-door surface (ONE-DOOR).

These tests run the real handler end-to-end with deterministic vectors. Before
the fix they fail with a "No module named 'semantic_kinematics.prompt_geometry'"
error surfaced inside the result dict; after the fix they produce a real drift.
"""

import asyncio

import numpy as np
import pytest

from semantic_kinematics.embeddings.base import EmbeddingAdapter
from semantic_kinematics.mcp.commands import embeddings


class _StubAdapter(EmbeddingAdapter):
    """Minimal concrete adapter so the *real* inherited cosine_distance runs."""

    @property
    def model_name(self) -> str:
        return "stub"

    @property
    def dimensions(self) -> int:
        return 3

    def embed(self, text: str) -> np.ndarray:  # pragma: no cover - unused here
        raise AssertionError("embed should be served via the manager's embed_fn")


class _FakeManager:
    """Serves both one-door surfaces calculate_drift depends on:
    get_embed_fn() (deterministic vectors) and get_adapter() (real metric)."""

    def __init__(self, vectors):
        self._vectors = vectors
        self._adapter = _StubAdapter()

    def get_embed_fn(self):
        return lambda text: self._vectors[text]

    def get_adapter(self):
        return self._adapter


def _run(coro):
    return asyncio.run(coro)


def test_calculate_drift_imports_and_runs_identical_vectors():
    """The dead-import bug surfaced as an error in the result; a clean numeric
    drift on identical vectors proves the import path is live."""
    v = np.array([1.0, 0.0, 0.0])
    mgr = _FakeManager({"alpha": v, "beta": v})

    result = _run(embeddings.calculate_drift(mgr, {"text_a": "alpha", "text_b": "beta"}))

    assert "error" not in result, result
    assert result["drift"] == pytest.approx(0.0, abs=1e-6)
    assert "similar" in result["interpretation"].lower()


def test_calculate_drift_orthogonal_vectors():
    mgr = _FakeManager(
        {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0])}
    )

    result = _run(embeddings.calculate_drift(mgr, {"text_a": "x", "text_b": "y"}))

    assert "error" not in result, result
    assert result["drift"] == pytest.approx(1.0, abs=1e-6)


def test_calculate_drift_requires_both_texts():
    mgr = _FakeManager({})
    result = _run(embeddings.calculate_drift(mgr, {"text_a": "only-a"}))
    assert "error" in result
    assert "required" in result["error"].lower()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
