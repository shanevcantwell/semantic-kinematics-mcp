"""Contract tests for the external TVI canonical-query stream adapter."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import scripts.tvi_canonical_input as canonical


def _metadata(
    *,
    returned_count: int,
    next_page_token: str | None,
    generation: str | None = None,
    projection: str = canonical.CHUNK_PROJECTION_VERSION,
    verb: str = canonical.CHUNK_QUERY_VERB,
    ordering: str = canonical.CHUNK_ORDERING,
    fingerprint: str = "5" * 64,
) -> dict:
    return {
        "record_type": "metadata",
        "metadata": {
            "query_contract_version": canonical.QUERY_CONTRACT_VERSION,
            "projection_version": projection,
            "provenance": {
                "canonical_contract_version": "tvi-canonical-corpus/1.0",
                "sqlite_schema_version": "tvi-canonical-corpus-sqlite/1.0",
                "generation_id": generation or f"tvi-generation-v1:sha256:{'1' * 64}",
                "database_sha256": "2" * 64,
                "logical_digest_sha256": "3" * 64,
                "verification_report_id": f"tvi-verification-v1:sha256:{'4' * 64}",
                "verification_report_contract_version": (
                    "tvi-canonical-corpus-verification-report/1.0"
                ),
                "verification_report_verified_at": "2026-10-27T23:32:10Z",
                "verified_open": True,
            },
            "query": {
                "verb": verb,
                "ordering": ordering,
                "query_fingerprint_sha256": fingerprint,
            },
            "pagination": {
                "limit": 10_000,
                "returned_count": returned_count,
                "next_page_token": next_page_token,
            },
        },
    }


def _row(chunk_id: str, text: str) -> dict:
    return {"record_type": "row", "row": {"chunk_id": chunk_id, "text": text}}


def _json_lines(records: list[dict]) -> list[str]:
    return [json.dumps(record, separators=(",", ":")) + "\n" for record in records]


def _fake_executable(
    tmp_path: Path,
    lines: list[str],
    *,
    exit_code: int = 0,
    stderr: str = "",
    name: str = "tvi-canonical-corpus",
) -> tuple[Path, Path]:
    invocation_log = tmp_path / f"{name}.invocations"
    script = tmp_path / name
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import json\n"
        "import sys\n"
        f"with open({str(invocation_log)!r}, 'a', encoding='utf-8') as log:\n"
        "    log.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"for line in {lines!r}:\n"
        "    sys.stdout.write(line)\n"
        "    sys.stdout.flush()\n"
        f"sys.stderr.write({stderr!r})\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | 0o111)
    return script, invocation_log


def _invocations(path: Path) -> list[list[str]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_build_chunks_argv_uses_explicit_all_pages_query_arguments(tmp_path: Path) -> None:
    argv = canonical.build_chunks_argv(
        tmp_path / "corpus.db",
        canonical_root=tmp_path / "canonical",
        input_keys=("base", "delta"),
        channels=("claude_code",),
        speakers=("user", "assistant"),
        conversation_selectors=({"acquisition_id": "acq-1", "conversation_id": "conversation-1"},),
        all_acquisitions_conversation_ids=("legacy-conversation",),
        nonblank_text=True,
        time_roles=("event_time",),
        executable="/opt/bin/tvi-canonical-corpus",
    )

    assert argv == [
        "/opt/bin/tvi-canonical-corpus",
        "query",
        "chunks",
        "--db",
        str(tmp_path / "corpus.db"),
        "--canonical-root",
        str(tmp_path / "canonical"),
        "--format",
        "jsonl",
        "--all-pages",
        "--limit",
        "10000",
        "--input-key",
        "base",
        "--input-key",
        "delta",
        "--channel",
        "claude_code",
        "--speaker",
        "user",
        "--speaker",
        "assistant",
        "--conversation-selector",
        '{"acquisition_id":"acq-1","conversation_id":"conversation-1"}',
        "--all-acquisitions-conversation-id",
        "legacy-conversation",
        "--nonblank-text",
        "--time-role",
        "event_time",
    ]


def test_stream_invokes_executable_once_without_shell_and_exposes_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [_metadata(returned_count=2, next_page_token=None), _row("a", "A"), _row("b", "B")]
    executable, invocation_log = _fake_executable(tmp_path, _json_lines(records))
    real_popen = subprocess.Popen
    calls: list[tuple[list[str], dict]] = []

    def recording_popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return real_popen(argv, **kwargs)

    monkeypatch.setattr(canonical.subprocess, "Popen", recording_popen)

    stream = canonical.open_canonical_chunks(
        tmp_path / "corpus.db",
        canonical_root=tmp_path,
        input_keys=("base",),
        executable=executable,
    )
    assert stream.provenance.generation_id.startswith("tvi-generation-v1:sha256:")
    assert stream.provenance.verified_open is True
    assert stream.provenance.as_dict()["query"] == {
        "verb": "iter_legacy_chunks",
        "ordering": "whole_corpus_v1",
        "query_fingerprint_sha256": "5" * 64,
    }
    assert [row["chunk_id"] for row in stream] == ["a", "b"]

    assert len(calls) == 1
    assert isinstance(calls[0][0], list)
    assert calls[0][1]["shell"] is False
    assert _invocations(invocation_log) == [calls[0][0][1:]]


def test_multi_page_stream_yields_rows_in_query_door_order(tmp_path: Path) -> None:
    records = [
        _metadata(returned_count=2, next_page_token="page-2"),
        _row("a", "A"),
        _row("b", "B"),
        _metadata(returned_count=1, next_page_token=None),
        _row("c", "C"),
    ]
    executable, invocation_log = _fake_executable(tmp_path, _json_lines(records))

    with canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable) as stream:
        assert [(row["chunk_id"], row["text"]) for row in stream] == [
            ("a", "A"),
            ("b", "B"),
            ("c", "C"),
        ]

    assert len(_invocations(invocation_log)) == 1


@pytest.mark.parametrize(
    "replacement",
    [
        {"generation": f"tvi-generation-v1:sha256:{'9' * 64}"},
        {"projection": "tvi-canonical-query-other/1.0"},
        {"verb": "lookup_legacy_chunks"},
        {"ordering": "conversation_v1"},
        {"fingerprint": "8" * 64},
    ],
    ids=("generation", "projection", "query", "ordering", "fingerprint"),
)
def test_cross_page_identity_changes_are_fatal(tmp_path: Path, replacement: dict) -> None:
    records = [
        _metadata(returned_count=1, next_page_token="page-2"),
        _row("a", "A"),
        _metadata(returned_count=1, next_page_token=None, **replacement),
        _row("b", "B"),
    ]
    executable, _ = _fake_executable(tmp_path, _json_lines(records))

    with pytest.raises(canonical.CanonicalProtocolError):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))


def test_repeated_page_token_is_fatal(tmp_path: Path) -> None:
    records = [
        _metadata(returned_count=1, next_page_token="page-2"),
        _row("a", "A"),
        _metadata(returned_count=1, next_page_token="page-2"),
        _row("b", "B"),
    ]
    executable, _ = _fake_executable(tmp_path, _json_lines(records))

    with pytest.raises(canonical.CanonicalProtocolError, match="repeated a next_page_token"):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))


def test_nonterminal_zero_row_page_is_fatal(tmp_path: Path) -> None:
    records = [_metadata(returned_count=0, next_page_token="page-2")]
    executable, _ = _fake_executable(tmp_path, _json_lines(records))

    with pytest.raises(canonical.CanonicalProtocolError, match="must emit at least one row"):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))


def test_duplicate_chunk_id_across_pages_is_fatal(tmp_path: Path) -> None:
    records = [
        _metadata(returned_count=1, next_page_token="page-2"),
        _row("a", "A"),
        _metadata(returned_count=1, next_page_token=None),
        _row("a", "replayed"),
    ]
    executable, _ = _fake_executable(tmp_path, _json_lines(records))

    with pytest.raises(canonical.CanonicalProtocolError, match="repeated chunk_id 'a'"):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))


@pytest.mark.parametrize(
    "lines, message",
    [
        (["not-json\n"], "malformed JSON"),
        (_json_lines([_row("a", "A")]), "must be metadata"),
        ([], "no metadata record"),
        (
            _json_lines([_metadata(returned_count=1, next_page_token=None)]),
            "row count does not match",
        ),
        (
            _json_lines(
                [
                    _metadata(returned_count=1, next_page_token="page-2"),
                    _row("a", "A"),
                ]
            ),
            "ended before the advertised next page",
        ),
        (
            _json_lines([_metadata(returned_count=1, next_page_token=None)])
            + [json.dumps(_row("a", "A"))],
            "not newline-terminated",
        ),
    ],
    ids=(
        "malformed-json",
        "row-before-metadata",
        "missing-metadata",
        "short-page",
        "missing-next-page",
        "truncated-line",
    ),
)
def test_malformed_or_broken_stream_is_refused(
    tmp_path: Path, lines: list[str], message: str
) -> None:
    executable, _ = _fake_executable(tmp_path, lines)

    with pytest.raises(canonical.CanonicalProtocolError, match=message):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))


def test_nonzero_query_exit_is_fatal_and_never_retried(tmp_path: Path) -> None:
    executable, invocation_log = _fake_executable(
        tmp_path,
        [],
        exit_code=4,
        stderr="verified open failed",
    )

    with pytest.raises(
        canonical.CanonicalQueryError,
        match="exited with status 4: verified open failed",
    ):
        list(canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable))

    assert len(_invocations(invocation_log)) == 1


def test_rows_are_available_before_the_executable_finishes(tmp_path: Path) -> None:
    marker = tmp_path / "finished"
    executable = tmp_path / "tvi-canonical-corpus-incremental"
    metadata_line, row_line = _json_lines(
        [_metadata(returned_count=1, next_page_token=None), _row("a", "A")]
    )
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib\n"
        "import sys\n"
        "import time\n"
        f"sys.stdout.write({metadata_line!r} + {row_line!r})\n"
        "sys.stdout.flush()\n"
        "time.sleep(10)\n"
        f"pathlib.Path({str(marker)!r}).write_text('finished', encoding='utf-8')\n",
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | 0o111)

    stream = canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable)
    iterator = iter(stream)
    assert next(iterator) == {"chunk_id": "a", "text": "A"}
    assert not marker.exists()
    stream.close()


def test_closed_stream_never_restarts_the_executable(tmp_path: Path) -> None:
    records = [_metadata(returned_count=0, next_page_token=None)]
    executable, invocation_log = _fake_executable(tmp_path, _json_lines(records))
    stream = canonical.open_canonical_chunks(tmp_path / "corpus.db", executable=executable)

    assert stream.provenance.verified_open is True
    stream.close()
    with pytest.raises(canonical.CanonicalInputError, match="is closed"):
        _ = stream.provenance

    assert len(_invocations(invocation_log)) == 1


def test_filter_validation_fails_before_starting_a_process(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        canonical.open_canonical_chunks(
            tmp_path / "corpus.db",
            nonblank_text=True,
            blank_text=True,
        )

    with pytest.raises(ValueError, match="positive integer"):
        canonical.open_canonical_chunks(tmp_path / "corpus.db", limit=0)


def test_adapter_has_no_tvi_python_or_sqlite_dependency() -> None:
    source = Path(canonical.__file__).read_text(encoding="utf-8")
    assert "thought_vault_integration" not in source
    assert "import sqlite" not in source
    assert "sqlite3" not in source
    assert "shell=True" not in source
