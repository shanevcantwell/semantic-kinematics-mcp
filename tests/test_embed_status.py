"""Tests for embed_status.compute_status — the truthful run-completion signal.

The run wrapper trusts this (not raw line count / exit code) to decide DONE vs
keep-retrying, so its counting must be exact: done = distinct valid embeddings,
failed = ids present only as _failed markers, pending = total - done, and a later
success must supersede an earlier _failed marker.
"""

from __future__ import annotations

import json

from scripts.embed_status import compute_status


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_counts_done_failed_pending(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"chunk_id": "a", "text": "alpha"},
            {"chunk_id": "b", "text": "beta"},
            {"chunk_id": "c", "text": "gamma"},
            {"chunk_id": "blank", "text": "   "},  # skipped by _read_items
        ],
    )
    ckpt = tmp_path / "ckpt.jsonl"
    _write_jsonl(
        ckpt,
        [
            {"chunk_id": "a", "embedding": [0.1, 0.2]},  # done
            {"chunk_id": "b", "embedding": [0.0, 0.0], "_failed": True},  # failed
            # c absent -> pending
        ],
    )

    done, failed, pending, total = compute_status(str(corpus), str(ckpt))
    assert (done, failed, pending, total) == (1, 1, 2, 3)


def test_later_success_supersedes_failed_marker(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    ckpt = tmp_path / "ckpt.jsonl"
    _write_jsonl(
        ckpt,
        [
            {"chunk_id": "a", "embedding": [0.0], "_failed": True},  # earlier failure
            {"chunk_id": "a", "embedding": [0.5]},  # later retry succeeded
        ],
    )

    done, failed, pending, total = compute_status(str(corpus), str(ckpt))
    assert (done, failed, pending, total) == (1, 0, 0, 1)


def test_missing_checkpoint_is_all_pending(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}])

    done, failed, pending, total = compute_status(str(corpus), str(tmp_path / "absent.jsonl"))
    assert (done, failed, pending, total) == (0, 0, 2, 2)
