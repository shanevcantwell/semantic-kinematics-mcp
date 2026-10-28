"""Tests for embed_status.compute_status — the truthful run-completion signal.

The run wrapper trusts this (not raw line count / exit code) to decide DONE vs
keep-retrying, so its counting must be exact: done = distinct valid embeddings,
failed = ids present only as _failed markers, pending = total - done, and a later
success must supersede an earlier _failed marker.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import embed_corpus
from scripts.embed_status import compute_status


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _write_sourced_checkpoint(path, rows, source, *, dimensions=None):
    embed_corpus.reconcile_source_sidecar(str(path), source.provenance)
    _write_jsonl(path, rows)
    if dimensions is None:
        dimensions = next(
            len(row["embedding"]) for row in rows if isinstance(row.get("embedding"), list)
        )
    path.with_name(path.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": dimensions}),
        encoding="utf-8",
    )


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
    _write_sourced_checkpoint(
        ckpt,
        [
            {"chunk_id": "a", "embedding": [0.1, 0.2]},  # done
            {"chunk_id": "b", "embedding": [0.0, 0.0], "_failed": True},  # failed
            # c absent -> pending
        ],
        embed_corpus.load_corpus_source(corpus=str(corpus)),
    )

    done, failed, pending, total = compute_status(str(corpus), str(ckpt))
    assert (done, failed, pending, total) == (1, 1, 2, 3)


def test_later_success_supersedes_failed_marker(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    ckpt = tmp_path / "ckpt.jsonl"
    _write_sourced_checkpoint(
        ckpt,
        [
            {"chunk_id": "a", "embedding": [0.0], "_failed": True},  # earlier failure
            {"chunk_id": "a", "embedding": [0.5]},  # later retry succeeded
        ],
        embed_corpus.load_corpus_source(corpus=str(corpus)),
    )

    done, failed, pending, total = compute_status(str(corpus), str(ckpt))
    assert (done, failed, pending, total) == (1, 0, 0, 1)


@pytest.mark.parametrize("checkpoint_state", ["absent", "empty"])
def test_absent_or_empty_checkpoint_is_all_pending(tmp_path, checkpoint_state):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "x"}, {"chunk_id": "b", "text": "y"}])
    checkpoint = tmp_path / "checkpoint.jsonl"
    if checkpoint_state == "empty":
        checkpoint.touch()

    done, failed, pending, total = compute_status(str(corpus), str(checkpoint))
    assert (done, failed, pending, total) == (0, 0, 2, 2)


@pytest.mark.parametrize(
    ("name", "embedding", "failed"),
    [
        ("unmarked zero", [0.0, 0.0], False),
        ("wrong dimension", [1.0], False),
        ("nonnumeric", ["bad", "data"], False),
        ("nested", [[1.0, 2.0]], False),
        ("NaN", [float("nan"), 1.0], False),
        ("Inf", [float("inf"), 1.0], False),
        ("_failed", [1.0, 2.0], True),
        ("valid", ["1.0", "2.0"], False),
    ],
)
def test_invalid_vector_regression_matrix(tmp_path, name, embedding, failed):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "item", "text": "text"}])
    checkpoint = tmp_path / "checkpoint.jsonl"
    row = {"chunk_id": "item", "embedding": embedding}
    if failed:
        row["_failed"] = True
    _write_sourced_checkpoint(
        checkpoint,
        [row],
        embed_corpus.load_corpus_source(corpus=str(corpus)),
        dimensions=2,
    )

    status = compute_status(str(corpus), str(checkpoint))
    if name == "valid":
        assert status == (1, 0, 0, 1)
    elif name == "_failed":
        assert status == (0, 1, 1, 1)
    else:
        assert status == (0, 0, 1, 1)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        "{ not json",
        json.dumps({"model_name": "test-model", "dimensions": 0}),
        json.dumps({"model_name": "test-model", "dimensions": "2"}),
    ],
)
def test_populated_checkpoint_requires_valid_positive_dimension_metadata(tmp_path, metadata):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "checkpoint.jsonl"
    source = embed_corpus.load_corpus_source(corpus=str(corpus))
    embed_corpus.reconcile_source_sidecar(str(checkpoint), source.provenance)
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.5, 0.5]}])
    if metadata is not None:
        checkpoint.with_name(checkpoint.name + ".meta.json").write_text(metadata, encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        compute_status(str(corpus), str(checkpoint))


class _CanonicalStream:
    def __init__(self, rows):
        self.rows = rows
        self.provenance = SimpleNamespace(
            as_dict=lambda: {
                "query_contract_version": "tvi-canonical-query/1.0",
                "projection_version": "tvi-canonical-query-legacy-chunk/1.0",
                "generation_id": f"tvi-generation-v1:sha256:{'1' * 64}",
                "database_sha256": "2" * 64,
                "logical_digest_sha256": "3" * 64,
                "verification_report_id": f"tvi-verification-v1:sha256:{'4' * 64}",
                "verified_open": True,
                "query": {
                    "verb": "iter_legacy_chunks",
                    "ordering": "whole_corpus_v1",
                    "query_fingerprint_sha256": "5" * 64,
                },
            }
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter(self.rows)


def test_canonical_db_total_equals_equivalent_jsonl_total(tmp_path, monkeypatch):
    rows = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "blank", "text": " \u2003 "},
        {"chunk_id": "b", "text": "beta"},
    ]
    monkeypatch.setattr(
        embed_corpus,
        "open_canonical_chunks",
        lambda *args, **kwargs: _CanonicalStream(rows),
    )
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, rows)
    checkpoint_rows = [
        {"chunk_id": "a", "embedding": [0.0], "_failed": True},
        {"chunk_id": "a", "embedding": [0.5]},
        {"chunk_id": "b", "embedding": [0.0], "_failed": True},
    ]
    jsonl_checkpoint = tmp_path / "jsonl-checkpoint.jsonl"
    _write_sourced_checkpoint(
        jsonl_checkpoint,
        checkpoint_rows,
        embed_corpus.load_corpus_source(corpus=str(corpus)),
    )
    canonical_checkpoint = tmp_path / "canonical-checkpoint.jsonl"
    _write_sourced_checkpoint(
        canonical_checkpoint,
        checkpoint_rows,
        embed_corpus.load_corpus_source(
            corpus_db=str(tmp_path / "corpus.db"), input_keys=("base",)
        ),
    )

    jsonl_status = compute_status(str(corpus), str(jsonl_checkpoint))
    canonical_status = compute_status(
        None,
        str(canonical_checkpoint),
        corpus_db=str(tmp_path / "corpus.db"),
        input_keys=("base",),
    )

    assert canonical_status == jsonl_status == (1, 1, 1, 2)


def test_extra_checkpoint_id_cannot_replace_missing_source_id(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"chunk_id": "a", "text": "alpha"},
            {"chunk_id": "b", "text": "beta"},
        ],
    )
    checkpoint = tmp_path / "checkpoint.jsonl"
    _write_sourced_checkpoint(
        checkpoint,
        [
            {"chunk_id": "a", "embedding": [0.5]},
            {"chunk_id": "stale", "embedding": [0.7]},
        ],
        embed_corpus.load_corpus_source(corpus=str(corpus)),
    )

    with pytest.raises(ValueError, match="absent from the current source"):
        compute_status(str(corpus), str(checkpoint))


def test_duplicate_source_ids_are_refused(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(
        corpus,
        [
            {"chunk_id": "a", "text": "alpha"},
            {"chunk_id": "a", "text": "again"},
        ],
    )

    with pytest.raises(ValueError, match="duplicate chunk_id"):
        compute_status(str(corpus), str(tmp_path / "checkpoint.jsonl"))


def test_status_refuses_checkpoint_source_mismatch(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "checkpoint.jsonl"
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.5]}])
    sidecar = tmp_path / "checkpoint.jsonl.source.json"
    sidecar.write_text(
        json.dumps(
            {
                "source_kind": "legacy_jsonl",
                "path": str(corpus.resolve()),
                "sha256": "0" * 64,
                "text_field": "text",
                "id_field": "chunk_id",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source mismatch"):
        compute_status(str(corpus), str(checkpoint))
