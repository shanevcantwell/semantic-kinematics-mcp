"""Canonical corpus input parity tests for both measured-null builders.

These tests exercise the shared TVI CLI stream with a fake executable. They do
not construct a real embedding adapter or make network requests.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.build_conditioned_null as conditioned
import scripts.build_displacement_null as displacement
import scripts.tvi_canonical_input as canonical

BUILDERS = (displacement, conditioned)


def _metadata(*, returned_count: int) -> dict:
    return {
        "record_type": "metadata",
        "metadata": {
            "query_contract_version": canonical.QUERY_CONTRACT_VERSION,
            "projection_version": canonical.CHUNK_PROJECTION_VERSION,
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
                "verification_report_verified_at": "2026-10-27T23:32:10Z",
                "verified_open": True,
            },
            "query": {
                "verb": canonical.CHUNK_QUERY_VERB,
                "ordering": canonical.CHUNK_ORDERING,
                "query_fingerprint_sha256": "5" * 64,
            },
            "pagination": {
                "limit": canonical.DEFAULT_PAGE_LIMIT,
                "returned_count": returned_count,
                "next_page_token": None,
            },
        },
    }


def _row(index: int, text: str) -> dict:
    return {"record_type": "row", "row": {"chunk_id": f"chunk-{index}", "text": text}}


def _fake_executable(
    tmp_path: Path,
    rows: list[dict],
    *,
    exit_code: int = 0,
    stderr: str = "",
    name: str = "tvi-canonical-corpus",
) -> tuple[Path, Path]:
    invocation_log = tmp_path / f"{name}.invocations"
    executable = tmp_path / name
    records = [_metadata(returned_count=len(rows)), *rows] if exit_code == 0 else []
    lines = [json.dumps(record, separators=(",", ":")) + "\n" for record in records]
    executable.write_text(
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
    executable.chmod(executable.stat().st_mode | 0o111)
    return executable, invocation_log


def _legacy_jsonl_rows(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield value


def _fixture_texts() -> list[str]:
    return [
        "",
        " \t\n ",
        "x" * 39,
        "  " + "alpha " * 8 + "  ",
        "\u2003" + "bravo " * 8 + "\u3000",
        "charlie " * 7,
        "delta " * 8,
        "echo " * 9,
        "foxtrot " * 7,
        "golf " * 10,
    ]


@pytest.mark.parametrize("builder", BUILDERS)
def test_jsonl_and_canonical_streams_produce_identical_usable_text_sequence(
    builder, tmp_path: Path
) -> None:
    texts = _fixture_texts()
    legacy_path = tmp_path / "chunks.jsonl"
    legacy_path.write_text(
        "\n".join(
            [json.dumps({"chunk_id": f"chunk-{i}", "text": text}) for i, text in enumerate(texts)]
            + ["not-json"]
        )
        + "\n",
        encoding="utf-8",
    )
    executable, invocation_log = _fake_executable(
        tmp_path, [_row(i, text) for i, text in enumerate(texts)]
    )

    legacy_usable = list(builder._iter_usable_texts(_legacy_jsonl_rows(legacy_path)))
    with canonical.open_canonical_chunks(
        tmp_path / "corpus.db",
        input_keys=("base",),
        executable=executable,
    ) as stream:
        canonical_usable = list(builder._iter_usable_texts(stream))

    assert canonical_usable == legacy_usable
    assert canonical_usable == [text.strip() for text in texts if len(text.strip()) >= 40]
    argv = json.loads(invocation_log.read_text(encoding="utf-8").splitlines()[0])
    assert argv[-2:] == ["--input-key", "base"]
    assert "--nonblank-text" not in argv


@pytest.mark.parametrize("builder", BUILDERS)
def test_fixed_seed_produces_identical_reservoir_sample_and_usable_count(
    builder, tmp_path: Path
) -> None:
    texts = _fixture_texts() + [f"usable row {i:02d} " * 4 for i in range(30)]
    legacy_path = tmp_path / "chunks.jsonl"
    legacy_path.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in texts),
        encoding="utf-8",
    )
    executable, _ = _fake_executable(
        tmp_path, [_row(i, text) for i, text in enumerate(texts)]
    )

    legacy_sample, legacy_count = builder._reservoir_sample_turns(
        _legacy_jsonl_rows(legacy_path), 7, 1729
    )
    with canonical.open_canonical_chunks(
        tmp_path / "corpus.db", input_keys=("base",), executable=executable
    ) as stream:
        canonical_sample, canonical_count = builder._reservoir_sample_turns(stream, 7, 1729)

    assert canonical_count == legacy_count
    assert canonical_sample == legacy_sample


@pytest.mark.parametrize("builder", BUILDERS)
def test_query_failure_stops_before_embedding_without_legacy_fallback(
    builder, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, invocation_log = _fake_executable(
        tmp_path,
        [],
        exit_code=4,
        stderr="verified open failed",
    )
    output = tmp_path / "must-not-exist.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(builder.__file__)),
            "--corpus-db",
            str(tmp_path / "corpus.db"),
            "--tvi-executable",
            str(executable),
            "--out",
            str(output),
        ],
    )

    assert builder.main() == 2
    assert not output.exists()
    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 1
    assert json.loads(invocations[0])[-2:] == ["--input-key", "base"]


def _assert_corpus_provenance(header: dict, consumer_name: str) -> None:
    provenance = header["corpus_provenance"]
    assert provenance["query_contract_version"] == canonical.QUERY_CONTRACT_VERSION
    assert provenance["projection_version"] == canonical.CHUNK_PROJECTION_VERSION
    assert provenance["generation_id"].startswith("tvi-generation-v1:sha256:")
    assert provenance["database_sha256"] == "2" * 64
    assert provenance["logical_digest_sha256"] == "3" * 64
    assert provenance["verification_report_id"].startswith("tvi-verification-v1:sha256:")
    assert provenance["verified_open"] is True
    assert provenance["query"] == {
        "verb": canonical.CHUNK_QUERY_VERB,
        "ordering": canonical.CHUNK_ORDERING,
        "query_fingerprint_sha256": "5" * 64,
    }
    assert provenance["normalized_input_keys"] == ["base"]
    assert provenance["normalized_filters"] == {
        "channels": [],
        "speakers": [],
        "conversations": [],
        "nonblank_text": None,
        "time_roles": [],
    }
    assert provenance["consumer"] == {"name": consumer_name, "version": "1.0"}
    assert provenance["produced_at"] == header["date"]


def test_displacement_main_writes_existing_schema_plus_corpus_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, _ = _fake_executable(tmp_path, [_row(0, "A useful sentence. Another useful sentence." * 2)])
    output = tmp_path / "displacement.json"

    class FakeAnalyzer:
        def tokenize_sentences(self, turn: str) -> list[str]:
            return ["first", "second"]

        def embed_sentences(self, sentences: list[str]) -> np.ndarray:
            value = 0.0 if sentences == ["first"] else 1.0
            return np.full((1, displacement.EXPECTED_DIM), value, dtype=float)

    monkeypatch.setattr(displacement, "_build_analyzer", lambda: FakeAnalyzer())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(displacement.__file__)),
            "--corpus-db",
            str(tmp_path / "corpus.db"),
            "--tvi-executable",
            str(executable),
            "--target-sentences",
            "2",
            "--out",
            str(output),
        ],
    )

    assert displacement.main() == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert set(artifact) == {"header", "stats", "magnitudes"}
    assert artifact["header"]["source_path"] == str(tmp_path / "corpus.db")
    _assert_corpus_provenance(artifact["header"], displacement.CONSUMER_NAME)


def test_conditioned_main_writes_existing_schema_plus_corpus_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable, _ = _fake_executable(tmp_path, [_row(0, "A useful phrase; another useful phrase." * 2)])
    output = tmp_path / "conditioned.json"

    phrases = [SimpleNamespace(content="first phrase"), SimpleNamespace(content="second phrase")]

    def fake_conditioned_vectors(_phrases, k, _adapter):
        matrix = np.zeros((2, conditioned.EXPECTED_DIM), dtype=float)
        matrix[1, 0] = float(k + 1)
        steps = [
            SimpleNamespace(actual_k=0, span_tokens=4, demarcator_class="NONE"),
            SimpleNamespace(actual_k=k, span_tokens=4, demarcator_class="TERM_FLOW"),
        ]
        return matrix, steps

    monkeypatch.setattr(conditioned, "LMStudioAdapter", lambda **kwargs: object())
    monkeypatch.setattr(conditioned, "segment", lambda turn: phrases)
    monkeypatch.setattr(conditioned, "conditioned_vectors", fake_conditioned_vectors)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path(conditioned.__file__)),
            "--corpus-db",
            str(tmp_path / "corpus.db"),
            "--tvi-executable",
            str(executable),
            "--turns",
            "1",
            "--out",
            str(output),
        ],
    )

    assert conditioned.main() == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert set(artifact) == {"header", "strata", "bias_diagnostics"}
    assert artifact["header"]["source_path"] == str(tmp_path / "corpus.db")
    _assert_corpus_provenance(artifact["header"], conditioned.CONSUMER_NAME)
