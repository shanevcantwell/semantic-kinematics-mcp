"""Tests for canonical text/ID totals plus vector-bound summary status."""

from __future__ import annotations

import builtins
import json
import os
from pathlib import Path

import pytest

from scripts import embed_corpus
from scripts.summarize_corpus import StoreSpec, _build_report, _human_size, _render
from scripts.tvi_canonical_input import CanonicalQueryError


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _record_source_sidecar(corpus, checkpoint):
    source = embed_corpus.load_corpus_source(corpus=str(corpus))
    embed_corpus.reconcile_source_sidecar(str(checkpoint), source.provenance)


def _metadata(returned_count: int) -> dict:
    return {
        "record_type": "metadata",
        "metadata": {
            "query_contract_version": "tvi-canonical-query/1.0",
            "projection_version": "tvi-canonical-query-legacy-chunk/1.0",
            "provenance": {
                "canonical_contract_version": "tvi-canonical-corpus/1.0",
                "sqlite_schema_version": "tvi-canonical-corpus-sqlite/1.0",
                "generation_id": f"tvi-generation-v1:sha256:{'1' * 64}",
                "database_sha256": "2" * 64,
                "logical_digest_sha256": "3" * 64,
                "verification_report_id": f"tvi-verification-v1:sha256:{'4' * 64}",
                "verification_report_contract_version": (
                    "tvi-canonical-corpus-verification-report/1.0"
                ),
                "verification_report_verified_at": "2026-10-28T00:00:00Z",
                "verified_open": True,
            },
            "query": {
                "verb": "iter_legacy_chunks",
                "ordering": "whole_corpus_v1",
                "query_fingerprint_sha256": "5" * 64,
            },
            "pagination": {
                "limit": 10_000,
                "returned_count": returned_count,
                "next_page_token": None,
            },
        },
    }


def _fake_canonical_executable(
    tmp_path: Path,
    rows: list[dict],
    *,
    exit_code: int = 0,
    stderr: str = "",
) -> Path:
    executable = tmp_path / "tvi-canonical-corpus"
    records = [_metadata(len(rows))] + [{"record_type": "row", "row": row} for row in rows]
    lines = [json.dumps(record) + "\n" for record in records]
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"for line in {lines!r}:\n"
        "    sys.stdout.write(line)\n"
        f"sys.stderr.write({stderr!r})\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    return executable


def _canonical_spec(tmp_path: Path, executable: Path, checkpoint: Path) -> StoreSpec:
    return StoreSpec(
        name="canonical-store",
        corpus_db=str(tmp_path / "corpus.db"),
        canonical_root=str(tmp_path),
        input_keys=("base",),
        tvi_executable=str(executable),
        checkpoint=str(checkpoint),
    )


def _record_canonical_sidecar(spec: StoreSpec):
    source = embed_corpus.load_corpus_source(
        corpus_db=spec.corpus_db,
        canonical_root=spec.canonical_root,
        input_keys=spec.input_keys,
        tvi_executable=spec.tvi_executable,
    )
    embed_corpus.reconcile_source_sidecar(spec.checkpoint, source.provenance)
    return source


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
    _record_source_sidecar(corpus, checkpoint)
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
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": 3}),
        encoding="utf-8",
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

    assert report.model_name == "test-model"
    assert report.dim == 3
    assert report.source_provenance["source_kind"] == "generic_jsonl"


def test_build_report_accepts_matching_legacy_jsonl_source_sidecar(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    source = embed_corpus.load_corpus_source(corpus=str(corpus))
    checkpoint = tmp_path / "ckpt.jsonl"
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1, 0.2]}])
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": 2}), encoding="utf-8"
    )
    checkpoint.with_name(checkpoint.name + ".source.json").write_text(
        json.dumps(dict(source.provenance, source_kind="legacy_jsonl")), encoding="utf-8"
    )

    report = _build_report(
        tmp_path,
        StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint)),
    )

    assert report.checkpoint_source_bound is True
    assert report.checkpoint_source_binding.startswith("validated ")
    assert "explicit generic JSONL" in _render([report])


def test_build_report_reads_meta_sidecar(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "ckpt.jsonl"
    _record_source_sidecar(corpus, checkpoint)
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1, 0.2]}])
    meta = tmp_path / "ckpt.jsonl.meta.json"
    meta.write_text(json.dumps({"model_name": "some-model", "dimensions": 2}), encoding="utf-8")

    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert report.model_name == "some-model"
    assert report.dim == 2


def test_build_report_uses_metadata_dimension_to_reject_invalid_vector(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "ckpt.jsonl"
    _record_source_sidecar(corpus, checkpoint)
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1]}])
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "some-model", "dimensions": 2}),
        encoding="utf-8",
    )

    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert (report.done, report.failed, report.pending, report.total) == (0, 0, 1, 1)
    assert report.model_name == "some-model"
    assert report.dim == 2
    assert report.completion_pct == 0.0


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


def test_build_report_observes_legacy_checkpoint_without_minting_sidecars(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "legacy.jsonl"
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1, 0.2]}])

    spec = StoreSpec(name="legacy-store", corpus=str(corpus), checkpoint=str(checkpoint))
    report = _build_report(tmp_path, spec)

    assert (report.done, report.failed, report.pending, report.total) == (1, 0, 0, 1)
    assert report.model_name == "(MISSING SIDECAR — tvi#37)"
    assert report.dim == 2
    assert report.checkpoint_metadata_status.startswith("legacy checkpoint")
    assert report.checkpoint_source_binding.startswith("legacy-unbound")
    assert not Path(str(checkpoint) + ".meta.json").exists()
    assert not Path(str(checkpoint) + ".source.json").exists()


def test_canonical_report_preserves_exact_total_ordered_ids_and_vector_status(
    tmp_path, monkeypatch
):
    rows = [
        {"chunk_id": "c", "text": "gamma"},
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "blank", "text": "\u2003"},
        {"chunk_id": "b", "text": "beta"},
    ]
    executable = _fake_canonical_executable(tmp_path, rows)
    checkpoint = tmp_path / "vectors.jsonl"
    spec = _canonical_spec(tmp_path, executable, checkpoint)
    source = _record_canonical_sidecar(spec)
    assert source.items == [("gamma", "c"), ("alpha", "a"), ("beta", "b")]
    _write_jsonl(
        checkpoint,
        [
            {"chunk_id": "c", "embedding": [0.1, 0.2]},
            {"chunk_id": "a", "embedding": [0.0, 0.0], "_failed": True},
            {"chunk_id": "a", "embedding": [0.3, 0.4]},
            {"chunk_id": "b", "embedding": [0.0, 0.0], "_failed": True},
        ],
    )
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": 2}),
        encoding="utf-8",
    )

    opened: list[str] = []
    real_open = builtins.open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    report = _build_report(tmp_path, spec)

    # Canonical rows c,a,b are the exact ordered/selected nonblank source set.
    assert (report.done, report.failed, report.pending, report.total) == (2, 1, 1, 3)
    assert report.raw_lines == 4
    assert report.distinct_ids == 3
    assert report.source_provenance["normalized_input_keys"] == ["base"]
    assert report.source_provenance["generation_id"].endswith("1" * 64)
    assert report.source_provenance["query"] == {
        "verb": "iter_legacy_chunks",
        "ordering": "whole_corpus_v1",
        "query_fingerprint_sha256": "5" * 64,
    }

    # Parent-process JSONL opens are exclusively the model-keyed checkpoint.
    jsonl_opens = [path for path in opened if path.endswith(".jsonl")]
    assert jsonl_opens
    assert set(jsonl_opens) == {str(checkpoint)}

    rendered = _render([report])
    assert "verified TVI canonical query" in rendered
    assert "Checkpoint SHA-256" in rendered
    assert report.checkpoint_sha256 in rendered
    assert "Checkpoint generation binding |" in rendered
    assert "Completion (source-bound checkpoint)" in rendered
    assert report.source_provenance["generation_id"] in rendered
    assert "5" * 64 in rendered
    assert f"**Query executable**: `{spec.tvi_executable}`" in rendered
    assert f"**Trusted canonical root**: `{spec.canonical_root}`" in rendered
    assert f"**Checkpoint (vector state)**: `{spec.checkpoint}`" in rendered


def test_atomic_checkpoint_replacement_fails_closed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "ckpt.jsonl"
    _record_source_sidecar(corpus, checkpoint)
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [0.1, 0.2]}])
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": 2}), encoding="utf-8"
    )
    replacement = tmp_path / "replacement.jsonl"
    _write_jsonl(replacement, [{"chunk_id": "a", "embedding": [0.3, 0.4]}])
    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))

    real_loads = json.loads
    replaced = False

    def replacing_loads(value, *args, **kwargs):
        nonlocal replaced
        result = real_loads(value, *args, **kwargs)
        if isinstance(value, bytes) and not replaced:
            replaced = True
            os.replace(replacement, checkpoint)
        return result

    monkeypatch.setattr(json, "loads", replacing_loads)
    with pytest.raises(ValueError, match="changed during report scan"):
        _build_report(tmp_path, spec)


def test_same_size_rewrite_with_restored_mtime_fails_closed(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    _write_jsonl(corpus, [{"chunk_id": "a", "text": "alpha"}])
    checkpoint = tmp_path / "ckpt.jsonl"
    _record_source_sidecar(corpus, checkpoint)
    original = json.dumps({"chunk_id": "a", "embedding": [0.1, 0.2]}) + "\n"
    replacement = json.dumps({"chunk_id": "a", "embedding": [0.3, 0.4]}) + "\n"
    assert len(original) == len(replacement)
    checkpoint.write_text(original, encoding="utf-8")
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "test-model", "dimensions": 2}), encoding="utf-8"
    )
    spec = StoreSpec(name="test-store", corpus=str(corpus), checkpoint=str(checkpoint))
    initial_stat = checkpoint.stat()

    real_loads = json.loads
    rewritten = False

    def rewriting_loads(value, *args, **kwargs):
        nonlocal rewritten
        result = real_loads(value, *args, **kwargs)
        if isinstance(value, bytes) and not rewritten:
            rewritten = True
            with checkpoint.open("r+b") as mutable:
                mutable.write(replacement.encode("utf-8"))
                mutable.flush()
                os.fsync(mutable.fileno())
            os.utime(
                checkpoint,
                ns=(initial_stat.st_atime_ns, initial_stat.st_mtime_ns),
            )
        return result

    monkeypatch.setattr(json, "loads", rewriting_loads)
    with pytest.raises(ValueError, match="changed during report scan"):
        _build_report(tmp_path, spec)


def test_canonical_verified_open_failure_precedes_all_checkpoint_reads(tmp_path, monkeypatch):
    executable = _fake_canonical_executable(
        tmp_path,
        [],
        exit_code=4,
        stderr="verified open failed",
    )
    checkpoint = tmp_path / "vectors.jsonl"
    _write_jsonl(checkpoint, [{"chunk_id": "a", "embedding": [1.0]}])
    spec = _canonical_spec(tmp_path, executable, checkpoint)

    opened: list[str] = []
    real_open = builtins.open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    with pytest.raises(CanonicalQueryError, match="verified open failed"):
        _build_report(tmp_path, spec)

    assert str(checkpoint) not in opened
    assert str(checkpoint) + ".meta.json" not in opened


def test_human_size_formats_units():
    assert _human_size(500) == "500.0 B"
    assert _human_size(2048) == "2.0 KB"
    assert _human_size(5 * 1024 * 1024) == "5.0 MB"
    assert _human_size(3 * 1024 * 1024 * 1024) == "3.0 GB"
