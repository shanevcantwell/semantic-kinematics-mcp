"""Gate-4 regression tests for smoke_jolt's canonical text/vector split."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import smoke_jolt
from scripts.tvi_canonical_input import CanonicalQueryError


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


def _fake_executable(tmp_path: Path, rows: list[dict]) -> tuple[Path, Path]:
    executable = tmp_path / "tvi-canonical-corpus"
    invocation_log = tmp_path / "invocations.jsonl"
    records = [_metadata(len(rows))] + [{"record_type": "row", "row": row} for row in rows]
    lines = [json.dumps(record) + "\n" for record in records]
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        f"with open({str(invocation_log)!r}, 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"for line in {lines!r}:\n"
        "    sys.stdout.write(line)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    return executable, invocation_log


def _provenance() -> SimpleNamespace:
    return SimpleNamespace(
        generation_id=f"tvi-generation-v1:sha256:{'1' * 64}",
        database_sha256="2" * 64,
        logical_digest_sha256="3" * 64,
        query_verb="iter_legacy_chunks",
        ordering="whole_corpus_v1",
        query_fingerprint_sha256="5" * 64,
    )


def _write_checkpoint_sidecars(checkpoint: Path, provenance=None) -> None:
    provenance = provenance or _provenance()
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps(
            {
                "model_name": smoke_jolt.EMBED_MODEL,
                "dimensions": smoke_jolt.CHECKPOINT_DIMENSIONS,
            }
        ),
        encoding="utf-8",
    )
    checkpoint.with_name(checkpoint.name + ".source.json").write_text(
        json.dumps(
            {
                "source_kind": "tvi_canonical_db",
                "generation_id": provenance.generation_id,
                "database_sha256": provenance.database_sha256,
                "logical_digest_sha256": provenance.logical_digest_sha256,
                "normalized_input_keys": ["base"],
                "normalized_filters": {
                    "channels": [],
                    "speakers": [],
                    "conversations": [],
                    "nonblank_text": None,
                    "time_roles": [],
                },
                "query": {
                    "verb": provenance.query_verb,
                    "ordering": provenance.ordering,
                    "query_fingerprint_sha256": "6" * 64,
                },
            }
        ),
        encoding="utf-8",
    )


def _row(
    chunk_id: str,
    *,
    conversation_id: str,
    speaker: str,
    message_index: int,
    text: str,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "conversation_id": conversation_id,
        "speaker": speaker,
        "message_index": message_index,
        "text": text,
    }


def test_specimen_b_exact_selection_order_count_and_query_provenance(tmp_path, monkeypatch):
    target = smoke_jolt.TARGET_CONVERSATION_ID
    rows = [
        _row(
            "assistant-2",
            conversation_id=target,
            speaker="assistant",
            message_index=2,
            text="assistant second",
        ),
        _row(
            "thinking-1",
            conversation_id=target,
            speaker="thinking",
            message_index=1,
            text="excluded thought",
        ),
        _row(
            "user-0",
            conversation_id=target,
            speaker="user",
            message_index=0,
            text="user first",
        ),
        _row(
            "system-3",
            conversation_id=target,
            speaker="system",
            message_index=3,
            text="excluded system",
        ),
        _row(
            "other-conversation",
            conversation_id="other",
            speaker="user",
            message_index=0,
            text="excluded conversation",
        ),
        _row(
            "assistant-2b",
            conversation_id=target,
            speaker="assistant",
            message_index=2,
            text="stable tie",
        ),
    ]
    executable, invocation_log = _fake_executable(tmp_path, rows)
    opened: list[str] = []
    real_open = builtins.open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(builtins, "open", recording_open)
        turns, provenance = smoke_jolt._load_specimen_b_turns(
            corpus_db=str(tmp_path / "corpus.db"),
            canonical_root=str(tmp_path),
            executable=str(executable),
        )

    assert not [path for path in opened if path.endswith(".jsonl")]
    assert turns == [
        ("user-0", "user first"),
        ("assistant-2", "assistant second"),
        ("assistant-2b", "stable tie"),
    ]
    assert len(turns) == 3
    assert provenance.generation_id.endswith("1" * 64)
    assert provenance.logical_digest_sha256 == "3" * 64
    assert provenance.query_verb == "iter_legacy_chunks"
    assert provenance.ordering == "whole_corpus_v1"
    assert provenance.query_fingerprint_sha256 == "5" * 64

    argv = json.loads(invocation_log.read_text(encoding="utf-8").splitlines()[0])
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--input-key"] == [
        "base"
    ]
    assert [argv[index + 1] for index, value in enumerate(argv) if value == "--speaker"] == [
        "user",
        "assistant",
    ]
    assert [
        argv[index + 1]
        for index, value in enumerate(argv)
        if value == "--all-acquisitions-conversation-id"
    ] == [target]
    assert "chunks.jsonl" not in " ".join(argv)


def test_checkpoint_read_is_retained_as_the_only_jsonl_vector_boundary(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embed_checkpoint.jsonl"
    vector = [0.1] * smoke_jolt.CHECKPOINT_DIMENSIONS
    checkpoint.write_text(
        json.dumps({"chunk_id": "a", "embedding": vector}) + "\n",
        encoding="utf-8",
    )
    provenance = _provenance()
    _write_checkpoint_sidecars(checkpoint, provenance)
    monkeypatch.setattr(smoke_jolt, "CHECKPOINT_PATH", str(checkpoint))

    opened: list[str] = []
    real_path_open = Path.open

    def recording_open(path, *args, **kwargs):
        opened.append(os.fspath(path))
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", recording_open)
    found, verification = smoke_jolt._load_embeddings_for(
        ["a"], turns=[("a", "alpha")], provenance=provenance
    )

    assert np.array_equal(found["a"], np.asarray(vector))
    assert verification.model_authoritative is True
    assert [path for path in opened if path.endswith(".jsonl")] == [str(checkpoint)]
    assert not hasattr(smoke_jolt, "CHUNKS_PATH")


def test_checkpoint_retry_resolution_ignores_failed_invalid_and_zero_vectors(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embed_checkpoint.jsonl"
    valid = [0.25] * smoke_jolt.CHECKPOINT_DIMENSIONS
    _write_checkpoint_sidecars(checkpoint)
    checkpoint.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in [
                {
                    "chunk_id": "a",
                    "embedding": [0.0] * smoke_jolt.CHECKPOINT_DIMENSIONS,
                    "_failed": True,
                },
                {"chunk_id": "a", "embedding": [0.0] * smoke_jolt.CHECKPOINT_DIMENSIONS},
                {"chunk_id": "a", "embedding": [float("nan")] + [1.0] * 767},
                {"chunk_id": "a", "embedding": valid},
                {
                    "chunk_id": "a",
                    "embedding": [0.0] * smoke_jolt.CHECKPOINT_DIMENSIONS,
                    "_failed": True,
                },
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke_jolt, "CHECKPOINT_PATH", str(checkpoint))

    found, _verification = smoke_jolt._load_embeddings_for(
        ["a"], turns=[("a", "alpha")], provenance=_provenance()
    )

    assert np.array_equal(found["a"], np.asarray(valid))


def test_sidecar_less_checkpoint_requires_exact_legacy_identity(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embed_checkpoint.jsonl"
    vector = [0.5] * smoke_jolt.CHECKPOINT_DIMENSIONS
    checkpoint.write_text(
        json.dumps({"chunk_id": "a", "embedding": vector}) + "\n",
        encoding="utf-8",
    )
    turns = [("a", "alpha")]
    provenance = _provenance()
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(smoke_jolt, "CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(smoke_jolt, "LEGACY_CHECKPOINT_SHA256", checkpoint_sha)
    monkeypatch.setattr(smoke_jolt, "LEGACY_CANONICAL_GENERATION_ID", provenance.generation_id)
    monkeypatch.setattr(
        smoke_jolt,
        "LEGACY_CANONICAL_QUERY_FINGERPRINT",
        provenance.query_fingerprint_sha256,
    )
    monkeypatch.setattr(
        smoke_jolt,
        "LEGACY_SPECIMEN_SEQUENCE_SHA256",
        smoke_jolt._sequence_sha256(turns),
    )

    found, verification = smoke_jolt._load_embeddings_for(["a"], turns=turns, provenance=provenance)
    assert np.array_equal(found["a"], np.asarray(vector))
    assert verification.model_authoritative is False
    assert "not authoritative" in verification.model_basis
    with pytest.raises(SystemExit, match="refusing the cross-specimen/shared-space verdict"):
        smoke_jolt._require_authoritative_shared_model(verification)

    provenance.generation_id = "different-generation"
    with pytest.raises(SystemExit, match="outside the pinned legacy exception"):
        smoke_jolt._load_embeddings_for(["a"], turns=turns, provenance=provenance)


def test_checkpoint_model_must_match_specimen_a(tmp_path, monkeypatch):
    checkpoint = tmp_path / "embed_checkpoint.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "chunk_id": "a",
                "embedding": [0.5] * smoke_jolt.CHECKPOINT_DIMENSIONS,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_checkpoint_sidecars(checkpoint)
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
        json.dumps({"model_name": "other-model", "dimensions": 768}),
        encoding="utf-8",
    )
    monkeypatch.setattr(smoke_jolt, "CHECKPOINT_PATH", str(checkpoint))

    with pytest.raises(SystemExit, match="does not match specimen A"):
        smoke_jolt._load_embeddings_for(["a"], turns=[("a", "alpha")], provenance=_provenance())


def test_canonical_failure_does_not_open_vector_checkpoint(tmp_path, monkeypatch):
    executable = tmp_path / "tvi-canonical-corpus"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('verified open failed')\n"
        "raise SystemExit(4)\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)
    checkpoint = tmp_path / "embed_checkpoint.jsonl"
    checkpoint.write_text("sentinel\n", encoding="utf-8")
    monkeypatch.setattr(smoke_jolt, "CHECKPOINT_PATH", str(checkpoint))

    opened: list[str] = []
    real_open = builtins.open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)):
            opened.append(os.fspath(path))
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    with pytest.raises(CanonicalQueryError, match="verified open failed"):
        smoke_jolt._load_specimen_b_turns(
            corpus_db=str(tmp_path / "corpus.db"),
            canonical_root=str(tmp_path),
            executable=str(executable),
        )

    assert str(checkpoint) not in opened
