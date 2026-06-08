"""Unit tests for BulkEmbedder. No real embedding server is contacted."""

import hashlib
import json
from typing import List

import numpy as np
import pytest

from semantic_kinematics.embeddings.base import EmbeddingAdapter
from semantic_kinematics.embeddings.bulk import BulkEmbedder

DIM = 8


def _unit_vector_for(text: str) -> np.ndarray:
    """Deterministic unit vector seeded from the text."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM)
    return vec / np.linalg.norm(vec)


class FakeAdapter(EmbeddingAdapter):
    """Deterministic, offline adapter that counts embed_batch calls."""

    def __init__(self, fail: bool = False):
        self.batch_calls = 0
        self.batch_sizes: List[int] = []
        self._fail = fail

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return DIM

    def embed(self, text: str) -> np.ndarray:
        return _unit_vector_for(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        if self._fail:
            raise RuntimeError("simulated backend failure")
        return np.array([_unit_vector_for(t) for t in texts])


def _is_unit(vec: np.ndarray) -> bool:
    return np.isclose(np.linalg.norm(vec), 1.0, atol=1e-6)


def test_short_text_not_split():
    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, max_tokens_per_chunk=1000)
    short = "A short sentence."
    assert embedder._split_text(short, 1000) == [short]

    result = embedder.embed_corpus([(short, "a")])
    assert "a" in result
    assert result["a"].shape == (DIM,)
    # Single sub-chunk: vector used as-is, equals the adapter's vector exactly.
    np.testing.assert_allclose(result["a"], _unit_vector_for(short))


def test_long_text_split_and_averaged():
    adapter = FakeAdapter()
    # max_tokens_per_chunk small so the text must split.
    embedder = BulkEmbedder(adapter, max_tokens_per_chunk=5, max_tokens_per_request=10_000)
    long_text = ". ".join(f"Sentence number {i} here" for i in range(20)) + "."

    sub_chunks = embedder._split_text(long_text, 5)
    assert len(sub_chunks) > 1

    result = embedder.embed_corpus([(long_text, "long")])
    assert "long" in result
    # Averaged then renormalized -> unit norm.
    assert _is_unit(result["long"])


def test_request_batching_respects_token_budget():
    adapter = FakeAdapter()
    # Each text ~ "x" * 40 -> 40//4 + 1 = 11 tokens. Budget 25 -> 2 texts/group.
    # 5 texts -> groups of [2, 2, 1] = 3 embed_batch calls.
    embedder = BulkEmbedder(adapter, max_tokens_per_request=25, max_tokens_per_chunk=1000)
    items = [("x" * 40, f"id{i}") for i in range(5)]

    result = embedder.embed_corpus(items)
    assert len(result) == 5
    assert adapter.batch_calls == 3
    assert adapter.batch_sizes == [2, 2, 1]


def test_resume_skips_valid_retries_failed_and_zero(tmp_path):
    checkpoint = tmp_path / "ckpt.jsonl"
    valid_vec = _unit_vector_for("valid-text").tolist()
    lines = [
        {"chunk_id": "valid", "embedding": valid_vec},
        {"chunk_id": "failed", "embedding": [0.0] * DIM, "_failed": True},
        {"chunk_id": "zero", "embedding": [0.0] * DIM},
    ]
    checkpoint.write_text("\n".join(json.dumps(x) for x in lines) + "\n")

    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, checkpoint_path=str(checkpoint))

    items = [
        ("valid-text", "valid"),
        ("failed-text", "failed"),
        ("zero-text", "zero"),
    ]
    result = embedder.embed_corpus(items)

    # All three present in output, but only the 2 retried ones hit the backend.
    assert set(result) == {"valid", "failed", "zero"}
    assert adapter.batch_calls == 1
    assert adapter.batch_sizes == [2]
    # The valid one is the checkpointed vector.
    np.testing.assert_allclose(result["valid"], np.asarray(valid_vec))


def test_group_failure_marks_failed_and_continues(tmp_path):
    checkpoint = tmp_path / "ckpt.jsonl"
    adapter = FakeAdapter(fail=True)
    embedder = BulkEmbedder(
        adapter,
        max_tokens_per_request=10_000,
        max_tokens_per_chunk=1000,
        checkpoint_path=str(checkpoint),
    )
    items = [("text one", "a"), ("text two", "b")]
    result = embedder.embed_corpus(items)

    # Group errored -> nothing returned, but no exception raised.
    assert result == {}
    # Both written to checkpoint as _failed with zero vectors.
    entries = [json.loads(l) for l in checkpoint.read_text().splitlines() if l]
    by_id = {e["chunk_id"]: e for e in entries}
    assert by_id["a"]["_failed"] is True
    assert by_id["b"]["_failed"] is True
    assert by_id["a"]["embedding"] == [0.0] * DIM


def test_failed_entries_retried_on_next_run(tmp_path):
    checkpoint = tmp_path / "ckpt.jsonl"
    items = [("text one", "a"), ("text two", "b")]

    # First run fails the whole group.
    BulkEmbedder(
        FakeAdapter(fail=True),
        max_tokens_per_request=10_000,
        checkpoint_path=str(checkpoint),
    ).embed_corpus(items)

    # Second run with a working adapter retries and succeeds.
    good = FakeAdapter()
    result = BulkEmbedder(
        good,
        max_tokens_per_request=10_000,
        checkpoint_path=str(checkpoint),
    ).embed_corpus(items)

    assert set(result) == {"a", "b"}
    assert good.batch_calls == 1


def test_all_ids_present_for_mixed_corpus():
    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, max_tokens_per_chunk=5, max_tokens_per_request=50)
    short = "Tiny."
    long_text = ". ".join(f"Sentence {i} with words" for i in range(15)) + "."
    items = [(short, "s"), (long_text, "l"), ("Another tiny one.", "t")]

    result = embedder.embed_corpus(items)
    assert set(result) == {"s", "l", "t"}
    for vec in result.values():
        assert vec.shape == (DIM,)


def test_split_never_empty_for_nonempty_input():
    embedder = BulkEmbedder(FakeAdapter(), max_tokens_per_chunk=2)
    # A single oversized wordless token must still hard-split, never empty.
    blob = "x" * 200
    chunks = embedder._split_text(blob, 2)
    assert len(chunks) > 1
    assert all(c for c in chunks)
    assert "".join(chunks) == blob


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
