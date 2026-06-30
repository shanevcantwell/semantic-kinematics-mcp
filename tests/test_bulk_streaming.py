"""Streaming-prep tests for BulkEmbedder.

embed_corpus used to prep (split/tokenize) EVERY pending item before the first
embed -- an unresumable head: a crash before any group embedded lost all that
prep with no on-disk breadcrumb. It now streams in windows (prep a window,
embed, checkpoint, advance), so the whole run reconstructs from the checkpoint.
These tests pin: (1) windowing does not change output, (2) embedding starts
after prepping just the first window, (3) a resume re-preps only the items not
already in the checkpoint.
"""

from __future__ import annotations

import hashlib
import json
from typing import List

import numpy as np

from semantic_kinematics.embeddings.base import EmbeddingAdapter
from semantic_kinematics.embeddings.bulk import BulkEmbedder

DIM = 8


def _unit(text: str) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    v = np.random.default_rng(seed).standard_normal(DIM)
    return v / np.linalg.norm(v)


class FakeAdapter(EmbeddingAdapter):
    def __init__(self):
        self.batch_calls = 0

    @property
    def model_name(self) -> str:
        return "fake"

    @property
    def dimensions(self) -> int:
        return DIM

    def embed(self, text: str) -> np.ndarray:
        return _unit(text)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self.batch_calls += 1
        return np.array([_unit(t) for t in texts])

    def count_tokens(self, text: str) -> int:
        return len(text) // 4 + 1


def _mk(adapter, **kw):
    kw.setdefault("max_tokens_per_chunk", 1000)
    kw.setdefault("max_tokens_per_request", 1000)
    return BulkEmbedder(adapter, **kw)


ITEMS = [(f"text number {i}", f"id{i}") for i in range(7)]


def test_windowing_does_not_change_output():
    """A small prep_window must produce the same vectors as one big window."""
    whole = _mk(FakeAdapter(), prep_window=1000).embed_corpus(ITEMS)
    windowed = _mk(FakeAdapter(), prep_window=2).embed_corpus(ITEMS)

    assert set(whole) == set(windowed) == {cid for _, cid in ITEMS}
    for cid in whole:
        assert np.allclose(whole[cid], windowed[cid])


def test_embeds_before_all_items_are_prepped(monkeypatch):
    """The first embed_batch happens after prepping only the first window,
    not the whole corpus -- the streaming property."""
    adapter = FakeAdapter()
    embedder = _mk(adapter, prep_window=2)

    split_count = {"n": 0}
    orig_split = embedder._split_text
    monkeypatch.setattr(
        embedder,
        "_split_text",
        lambda t, m: (split_count.__setitem__("n", split_count["n"] + 1) or orig_split(t, m)),
    )

    splits_seen_at_first_batch = {}
    orig_batch = adapter.embed_batch

    def spy_batch(texts):
        splits_seen_at_first_batch.setdefault("n", split_count["n"])
        return orig_batch(texts)

    monkeypatch.setattr(adapter, "embed_batch", spy_batch)

    embedder.embed_corpus(ITEMS)

    # Exactly the first window (2 items) had been split when embedding began;
    # the old prepare-everything-first code would show all 7.
    assert splits_seen_at_first_batch["n"] == 2


def test_resume_does_not_reprep_completed(tmp_path, monkeypatch):
    """A fully-completed corpus re-runs with zero prep and zero embed work."""
    ckpt = tmp_path / "ck.jsonl"
    _mk(FakeAdapter(), prep_window=2, checkpoint_path=str(ckpt)).embed_corpus(ITEMS)

    adapter = FakeAdapter()
    embedder = _mk(adapter, prep_window=2, checkpoint_path=str(ckpt))
    split_count = {"n": 0}
    orig = embedder._split_text
    monkeypatch.setattr(
        embedder,
        "_split_text",
        lambda t, m: (split_count.__setitem__("n", split_count["n"] + 1) or orig(t, m)),
    )

    result = embedder.embed_corpus(ITEMS)

    assert split_count["n"] == 0      # nothing re-prepped
    assert adapter.batch_calls == 0   # nothing re-embedded
    assert len(result) == len(ITEMS)  # all restored from checkpoint


def test_resume_preps_only_missing(tmp_path, monkeypatch):
    """A partial checkpoint -> prep covers only the not-yet-done items."""
    ckpt = tmp_path / "ck.jsonl"
    with open(ckpt, "w") as f:  # pre-seed 3 of 7 as done
        for i in range(3):
            f.write(json.dumps({"chunk_id": f"id{i}", "embedding": _unit(f"text number {i}").tolist()}) + "\n")

    adapter = FakeAdapter()
    embedder = _mk(adapter, prep_window=2, checkpoint_path=str(ckpt))
    split_count = {"n": 0}
    orig = embedder._split_text
    monkeypatch.setattr(
        embedder,
        "_split_text",
        lambda t, m: (split_count.__setitem__("n", split_count["n"] + 1) or orig(t, m)),
    )

    result = embedder.embed_corpus(ITEMS)

    assert split_count["n"] == len(ITEMS) - 3  # only the missing 4 re-prepped
    assert len(result) == len(ITEMS)


def test_all_prep_failures_window_flushes_failed_markers(tmp_path, monkeypatch):
    """A window where every item fails prep produces no groups; its _failed
    markers must still be written + flushed (durable for resume retry)."""
    ckpt = tmp_path / "ck.jsonl"
    adapter = FakeAdapter()
    embedder = _mk(adapter, prep_window=2, checkpoint_path=str(ckpt))

    def boom(text, mx):
        raise ValueError("prep boom")

    monkeypatch.setattr(embedder, "_split_text", boom)

    result = embedder.embed_corpus(ITEMS)

    assert result == {}             # nothing embedded
    assert adapter.batch_calls == 0  # no group ever formed
    written = [json.loads(line) for line in ckpt.read_text().splitlines() if line.strip()]
    failed_ids = {r["chunk_id"] for r in written if r.get("_failed")}
    assert failed_ids == {cid for _, cid in ITEMS}  # every item marked _failed, durably
