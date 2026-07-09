"""Tests for scripts/summarize_corpus.py — the derived-stats generator.

Builds a tiny temp corpus + checkpoint fixture with a known shape (a
_failed zero-vector row, and a duplicate chunk_id where a later successful
line supersedes an earlier one) and asserts the derived done/failed/pending/
total and the raw-vs-distinct delta match hand-computed expected values.
"""

from __future__ import annotations

import json

from scripts.summarize_corpus import StoreSpec, _build_report, _human_size


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_build_report_derives_expected_counts(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"chunk_id": "a", "text": "alpha"},
            {"chunk_id": "b", "text": "beta"},
            {"chunk_id": "c", "text": "gamma"},
            {"chunk_id": "d", "text": "delta"},
            {"chunk_id": "blank", "text": "   "},  # skipped -> not in total
        ],
    )
    checkpoint = tmp_path / "ckpt.jsonl"
    _write_jsonl(
        checkpoint,
        [
            {"chunk_id": "a", "embedding": [0.1, 0.2, 0.3]},  # done
            {"chunk_id": "b", "embedding": [0.0, 0.0, 0.0], "_failed": True},  # failed
            {"chunk_id": "c", "embedding": [0.0, 0.0, 0.0], "_failed": True},  # earlier failure
            {"chunk_id": "c", "embedding": [0.4, 0.5, 0.6]},  # later retry -> supersedes -> done
            # d absent -> pending
        ],
    )

    # total = 4 (blank skipped), done = {a, c} = 2, failed = {b} = 1
    # (c's earlier _failed marker is superseded by the later success)
    # pending = total - done = 4 - 2 = 2
    #
    # raw checkpoint lines = 4, distinct chunk_ids = {a, b, c} = 3
    # duplicate/retry lines = 4 - 3 = 1 (c's retry line)

    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert (report.done, report.failed, report.pending, report.total) == (2, 1, 2, 4)
    assert report.raw_lines == 4
    assert report.distinct_ids == 3
    assert report.duplicate_lines == 1
    assert report.completion_pct == 50.0

    # No .meta.json sidecar was written -> MISSING flag, dim inferred from
    # the first non-failed embedding (chunk_id "a", length 3).
    assert report.model_name == "(MISSING SIDECAR — tvi#37)"
    assert report.dim == 3


def test_build_report_reads_meta_sidecar(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "ckpt.jsonl"
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1, 0.2]}])
    meta = tmp_path / "ckpt.jsonl.meta.json"
    meta.write_text(json.dumps({"model_name": "some-model", "dimensions": 2}), encoding="utf-8")

    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert report.model_name == "some-model"
    assert report.dim == 2


def test_build_report_missing_checkpoint_is_all_pending(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}])
    checkpoint = tmp_path / "absent.jsonl"

    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert (report.done, report.failed, report.pending, report.total) == (0, 0, 2, 2)
    assert report.raw_lines == 0
    assert report.distinct_ids == 0
    assert report.checkpoint_exists is False
    assert report.model_name == "(MISSING SIDECAR — tvi#37)"
    assert report.dim is None


def test_human_size_formats_units():
    assert _human_size(500) == "500.0 B"
    assert _human_size(2048) == "2.0 KB"
    assert _human_size(5 * 1024 * 1024) == "5.0 MB"
    assert _human_size(3 * 1024 * 1024 * 1024) == "3.0 GB"
