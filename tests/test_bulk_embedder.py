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

    def count_tokens(self, text: str) -> int:
        """Deterministic offline token count.

        Mirrors the old chars/4 arithmetic so the batching tests stay legible,
        but it is now an explicit adapter method (the BulkEmbedder calls this,
        never a private heuristic) -- the contract issue #20 introduced.
        """
        return len(text) // 4 + 1


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
    # Single sub-chunk: L2-normalized on store; FakeAdapter already returns
    # unit vectors, so the result matches the adapter's vector.
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


class NonUnitAdapter(FakeAdapter):
    """Adapter that violates the unit-norm contract with NON-uniform scaling.

    Each returned vector in a batch is scaled by ``10 ** i`` for the i-th text,
    so per-chunk magnitudes differ by orders of magnitude. Uniform scaling
    (the old test's ``* 3.0``) preserves the mean's direction and so could not
    catch the magnitude-domination bug in issue #17; non-uniform scaling can.
    """

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        base = super().embed_batch(texts)
        scales = np.array([10.0 ** i for i in range(len(texts))]).reshape(-1, 1)
        return base * scales


def test_single_chunk_normalized_even_with_nonunit_adapter():
    adapter = NonUnitAdapter()
    embedder = BulkEmbedder(adapter, max_tokens_per_chunk=1000)
    result = embedder.embed_corpus([("A short sentence.", "a")])
    # k == 1 path must L2-normalize too, not store the adapter vector as-is.
    assert _is_unit(result["a"])


def test_multichunk_averaging_is_normalize_each_then_mean(monkeypatch):
    """#17: the reassembled direction must be the centroid of the *unit*
    sub-chunk vectors, independent of per-chunk magnitude.

    With a non-uniform-scaling adapter, averaging raw vectors would let the
    largest-norm sub-chunk dominate the direction. The fix normalizes each
    sub-chunk before averaging, so the result equals the direction a unit-norm
    adapter would have produced for the same sub-chunks.
    """
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four."

    # Force this text to split into multiple sub-chunks deterministically.
    nonunit = NonUnitAdapter()
    embedder = BulkEmbedder(nonunit, max_tokens_per_chunk=3, max_tokens_per_request=10_000)
    sub_chunks = embedder._split_text(text, 3)
    assert len(sub_chunks) > 1

    result = embedder.embed_corpus([(text, "x")])
    assert _is_unit(result["x"])

    # Expected: normalize each sub-chunk's UNIT vector (scale-invariant), mean,
    # normalize. The expected uses the unit vectors directly -- the scaling the
    # adapter applied must wash out.
    unit_chunks = np.array([_unit_vector_for(sc) for sc in sub_chunks])
    expected = unit_chunks.mean(axis=0)
    expected = expected / np.linalg.norm(expected)

    np.testing.assert_allclose(result["x"], expected, atol=1e-9)


def test_dense_text_splits_under_true_count_not_undershoot():
    """#20: a backend whose true token count is far above chars/4 must still
    get split so each sub-chunk fits the server's physical batch.

    The DenseAdapter reports 4x the chars/4 estimate -- mimicking dense
    code/JSON where chars/token runs ~1.1 instead of ~4. An input the old
    chars/4 heuristic believed fit the chunk ceiling now correctly splits,
    because BulkEmbedder asks the adapter for the true count.
    """

    class DenseAdapter(FakeAdapter):
        def count_tokens(self, text: str) -> int:
            # ~4x denser than chars/4: the regime that silently 500'd in #20.
            return (len(text) // 4 + 1) * 4

    adapter = DenseAdapter()
    # Chunk ceiling 400 true tokens. A 600-char text is ~150 by chars/4 (would
    # NOT split under the old heuristic) but ~600 true tokens here -> must split.
    embedder = BulkEmbedder(adapter, max_tokens_per_chunk=400, max_tokens_per_request=10_000)
    dense = ". ".join(f"token{i} dense fragment here" for i in range(25)) + "."

    # Sanity: the old chars/4 estimate is under the ceiling; the true count isn't.
    assert (len(dense) // 4 + 1) <= 400
    assert adapter.count_tokens(dense) > 400

    sub_chunks = embedder._split_text(dense, 400)
    assert len(sub_chunks) > 1
    # Every sub-chunk fits the true-token ceiling -> no oversized request.
    assert all(adapter.count_tokens(sc) <= 400 for sc in sub_chunks)

    result = embedder.embed_corpus([(dense, "dense")])
    assert "dense" in result
    assert _is_unit(result["dense"])


def test_no_pending_work_does_not_create_checkpoint(tmp_path):
    checkpoint = tmp_path / "ckpt.jsonl"
    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, checkpoint_path=str(checkpoint))
    result = embedder.embed_corpus([])
    assert result == {}
    assert adapter.batch_calls == 0
    # No pending work: the checkpoint file must not be created.
    assert not checkpoint.exists()


def test_no_pending_work_leaves_checkpoint_untouched(tmp_path):
    checkpoint = tmp_path / "ckpt.jsonl"
    valid_vec = _unit_vector_for("valid-text").tolist()
    original = json.dumps({"chunk_id": "valid", "embedding": valid_vec}) + "\n"
    checkpoint.write_text(original)

    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, checkpoint_path=str(checkpoint))
    result = embedder.embed_corpus([("valid-text", "valid")])

    # Everything already completed: no backend call, file byte-identical.
    assert set(result) == {"valid"}
    assert adapter.batch_calls == 0
    assert checkpoint.read_text() == original


def test_corrupt_checkpoint_line_keeps_valid_entries(tmp_path, capsys):
    checkpoint = tmp_path / "ckpt.jsonl"
    vec_a = _unit_vector_for("text-a").tolist()
    vec_c = _unit_vector_for("text-c").tolist()
    lines = [
        json.dumps({"chunk_id": "a", "embedding": vec_a}),
        "{this is not json",
        json.dumps({"chunk_id": "c", "embedding": vec_c}),
    ]
    checkpoint.write_text("\n".join(lines) + "\n")

    adapter = FakeAdapter()
    embedder = BulkEmbedder(adapter, checkpoint_path=str(checkpoint))
    items = [("text-a", "a"), ("text-b", "b"), ("text-c", "c")]
    result = embedder.embed_corpus(items)

    # Valid entries survive the corrupt middle line; only "b" is embedded.
    assert set(result) == {"a", "b", "c"}
    assert adapter.batch_calls == 1
    assert adapter.batch_sizes == [1]
    np.testing.assert_allclose(result["a"], np.asarray(vec_a))
    np.testing.assert_allclose(result["c"], np.asarray(vec_c))
    assert "corrupt checkpoint line 2" in capsys.readouterr().err


def test_over_budget_item_embeds_as_singleton_group(capsys):
    adapter = FakeAdapter()
    # Budget 10; "y" * 100 -> 100//4 + 1 = 26 tokens, over budget on its own.
    embedder = BulkEmbedder(adapter, max_tokens_per_request=10, max_tokens_per_chunk=1000)
    items = [("tiny", "t1"), ("y" * 100, "big"), ("tiny2", "t2")]

    result = embedder.embed_corpus(items)
    # The over-budget item still embeds, isolated in its own group.
    assert set(result) == {"t1", "big", "t2"}
    assert adapter.batch_sizes == [1, 1, 1]
    assert "over-budget" in capsys.readouterr().err


def test_split_never_empty_for_nonempty_input():
    embedder = BulkEmbedder(FakeAdapter(), max_tokens_per_chunk=2)
    # A single oversized wordless token must still hard-split, never empty.
    blob = "x" * 200
    chunks = embedder._split_text(blob, 2)
    assert len(chunks) > 1
    assert all(c for c in chunks)
    assert "".join(chunks) == blob


def test_hard_split_bisect_on_runon_input_terminates_under_ceiling():
    """_hard_split_by_tokens must bisect a single run-on input that has NO
    sentence-boundary punctuation into pieces each <= the ceiling, and the
    bisection must terminate.

    The DenseAdapter reports ~1 token/char so a punctuation-free run-on of
    plain characters has a true token count well over the ceiling; with no
    boundary to split on, _split_text falls through to _hard_split_by_tokens
    which bisects on character length against the real tokenizer.
    """

    class DenseAdapter(FakeAdapter):
        def count_tokens(self, text: str) -> int:
            # ~1 token/char: a punctuation-free blob blows past the ceiling.
            return max(1, len(text))

    adapter = DenseAdapter()
    ceiling = 20
    embedder = BulkEmbedder(
        adapter, max_tokens_per_chunk=ceiling, max_tokens_per_request=10_000
    )
    # No ., !, ?, or newline anywhere -> one atomic "sentence" far over ceiling.
    runon = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    assert adapter.count_tokens(runon) > ceiling

    pieces = embedder._hard_split_by_tokens(runon, ceiling)
    assert len(pieces) > 1
    # Every piece fits the true-token ceiling and the bisect terminated.
    assert all(adapter.count_tokens(p) <= ceiling for p in pieces)
    # Lossless reassembly.
    assert "".join(pieces) == runon

    # End-to-end through _split_text (no boundary -> hits the hard-split path).
    sub_chunks = embedder._split_text(runon, ceiling)
    assert len(sub_chunks) > 1
    assert all(adapter.count_tokens(sc) <= ceiling for sc in sub_chunks)

    result = embedder.embed_corpus([(runon, "runon")])
    assert "runon" in result
    assert _is_unit(result["runon"])


def test_count_tokens_failure_during_prep_marks_item_failed_and_continues(tmp_path):
    """Blocker gate: a count_tokens failure for ONE item during preparation
    must mark THAT item _failed in the checkpoint, still embed the others,
    write a resumable checkpoint, and NOT propagate the exception.

    Fails against the pre-fix code (prep ran outside the try, so the raise
    aborted the whole run with no _failed marker); passes after the fix.
    """

    class PrepFailAdapter(FakeAdapter):
        def count_tokens(self, text: str) -> int:
            if text == "boom":
                raise RuntimeError("simulated /tokenize failure")
            return len(text) // 4 + 1

    checkpoint = tmp_path / "ckpt.jsonl"
    adapter = PrepFailAdapter()
    embedder = BulkEmbedder(
        adapter,
        max_tokens_per_request=10_000,
        max_tokens_per_chunk=1000,
        checkpoint_path=str(checkpoint),
    )
    items = [("text one", "a"), ("boom", "bad"), ("text two", "b")]

    # No exception escapes.
    result = embedder.embed_corpus(items)

    # The two good items embedded; the failing one is omitted from output.
    assert set(result) == {"a", "b"}

    # The checkpoint records the failing item as _failed (resumable), and the
    # good items as completed -- already-done work is not lost.
    entries = [json.loads(l) for l in checkpoint.read_text().splitlines() if l]
    by_id = {e["chunk_id"]: e for e in entries}
    assert by_id["bad"]["_failed"] is True
    assert by_id["bad"]["embedding"] == [0.0] * DIM
    assert "_failed" not in by_id["a"]
    assert "_failed" not in by_id["b"]

    # Resume with a healthy adapter retries only the _failed item idempotently.
    good = FakeAdapter()
    result2 = BulkEmbedder(
        good,
        max_tokens_per_request=10_000,
        max_tokens_per_chunk=1000,
        checkpoint_path=str(checkpoint),
    ).embed_corpus([("recovered", "bad")] + [("text one", "a"), ("text two", "b")])
    assert "bad" in result2
    # Only the previously-failed item hit the backend on resume.
    assert good.batch_sizes == [1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
