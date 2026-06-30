"""Self-describing checkpoint tests (issue #16).

A bare {chunk_id, embedding} JSONL can't identify its producer, and the
dimension was the only resume guard -- two different models at the same dim
would silently merge. BulkEmbedder now writes a sidecar <checkpoint>.meta.json
(model_name + dimensions) and refuses to resume against a mismatched one.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from semantic_kinematics.embeddings.base import EmbeddingAdapter
from semantic_kinematics.embeddings.bulk import BulkEmbedder


def _unit(text: str, dim: int) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(dim)
    return v / np.linalg.norm(v)


class FakeAdapter(EmbeddingAdapter):
    def __init__(self, model_name: str = "fake-A", dim: int = 8):
        self._name = model_name
        self._dim = dim

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def dimensions(self) -> int:
        return self._dim

    def embed(self, text):
        return _unit(text, self._dim)

    def embed_batch(self, texts):
        return np.array([_unit(t, self._dim) for t in texts])

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


ITEMS = [(f"text {i}", f"id{i}") for i in range(5)]


def _embedder(adapter, ckpt):
    return BulkEmbedder(
        adapter, max_tokens_per_chunk=1000, max_tokens_per_request=1000, checkpoint_path=str(ckpt)
    )


def test_fresh_run_writes_self_describing_meta(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)

    meta = json.loads((tmp_path / "ck.jsonl.meta.json").read_text())
    assert meta == {"model_name": "fake-A", "dimensions": 8}


def test_matching_resume_is_allowed(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)
    # second run, same identity -> no error, all already complete
    result = _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)
    assert len(result) == len(ITEMS)


def test_resume_with_different_model_fails_loud(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)

    with pytest.raises(ValueError, match="different model"):
        _embedder(FakeAdapter("fake-B", 8), ckpt).embed_corpus(ITEMS)


def test_resume_with_different_dim_fails_loud(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)

    with pytest.raises(ValueError, match="different model"):
        _embedder(FakeAdapter("fake-A", 4), ckpt).embed_corpus(ITEMS)


def test_pre_issue16_checkpoint_without_meta_is_adopted(tmp_path):
    """A checkpoint that predates #16 (no sidecar) is adopted: the meta is
    written from the current adapter and the run proceeds (no false mismatch)."""
    ckpt = tmp_path / "ck.jsonl"
    # hand-write a valid old-format checkpoint, NO meta sidecar
    with open(ckpt, "w") as f:
        for _, cid in ITEMS[:2]:
            f.write(json.dumps({"chunk_id": cid, "embedding": _unit(cid, 8).tolist()}) + "\n")
    assert not (tmp_path / "ck.jsonl.meta.json").exists()

    result = _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)

    assert len(result) == len(ITEMS)
    assert json.loads((tmp_path / "ck.jsonl.meta.json").read_text()) == {
        "model_name": "fake-A",
        "dimensions": 8,
    }


def test_unreadable_meta_fails_loud(tmp_path):
    ckpt = tmp_path / "ck.jsonl"
    (tmp_path / "ck.jsonl.meta.json").write_text("{ not json")

    with pytest.raises(ValueError, match="unreadable"):
        _embedder(FakeAdapter("fake-A", 8), ckpt).embed_corpus(ITEMS)


def test_no_checkpoint_path_writes_no_meta(tmp_path):
    """With no checkpoint there is nothing to protect: no sidecar, no guard."""
    embedder = BulkEmbedder(
        FakeAdapter("fake-A", 8), max_tokens_per_chunk=1000, max_tokens_per_request=1000
    )
    result = embedder.embed_corpus(ITEMS)

    assert len(result) == len(ITEMS)
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere
