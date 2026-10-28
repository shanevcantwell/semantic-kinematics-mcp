#!/usr/bin/env python
"""Stream verified TVI canonical chunks through the installed query CLI.

This module intentionally depends only on the external ``tvi-canonical-corpus``
executable.  Importing TVI's Python package here would create a package cycle.
The executable is started exactly once for a traversal with ``--all-pages``;
rows are decoded from stdout as they arrive and are never read from SQLite.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TextIO, TypeVar

QUERY_CONTRACT_VERSION = "tvi-canonical-query/1.0"
CHUNK_PROJECTION_VERSION = "tvi-canonical-query-legacy-chunk/1.0"
CHUNK_QUERY_VERB = "iter_legacy_chunks"
CHUNK_ORDERING = "whole_corpus_v1"
DEFAULT_PAGE_LIMIT = 10_000
DEFAULT_EXECUTABLE = "tvi-canonical-corpus"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_StreamT = TypeVar("_StreamT", bound="CanonicalChunkStream")


class CanonicalInputError(RuntimeError):
    """Base class for canonical-input acquisition failures."""


class CanonicalQueryError(CanonicalInputError):
    """The installed TVI query executable could not complete successfully."""


class CanonicalProtocolError(CanonicalInputError):
    """The TVI query stream did not satisfy its JSONL contract."""


@dataclass(frozen=True)
class CanonicalQueryProvenance:
    """Verified identity and query description shared by every emitted page."""

    query_contract_version: str
    projection_version: str
    canonical_contract_version: str
    sqlite_schema_version: str
    generation_id: str
    database_sha256: str
    logical_digest_sha256: str
    verification_report_id: str
    verification_report_contract_version: str
    verification_report_verified_at: str
    verified_open: bool
    query_verb: str
    ordering: str
    query_fingerprint_sha256: str

    def as_dict(self) -> dict[str, Any]:
        """Return the normalized provenance shape expected by consumers."""
        return {
            "query_contract_version": self.query_contract_version,
            "projection_version": self.projection_version,
            "generation_id": self.generation_id,
            "database_sha256": self.database_sha256,
            "logical_digest_sha256": self.logical_digest_sha256,
            "verification_report_id": self.verification_report_id,
            "verified_open": self.verified_open,
            "query": {
                "verb": self.query_verb,
                "ordering": self.ordering,
                "query_fingerprint_sha256": self.query_fingerprint_sha256,
            },
            "canonical_contract_version": self.canonical_contract_version,
            "sqlite_schema_version": self.sqlite_schema_version,
            "verification_report_contract_version": self.verification_report_contract_version,
            "verification_report_verified_at": self.verification_report_verified_at,
        }


@dataclass
class _PageState:
    returned_count: int
    next_page_token: str | None
    rows_seen: int = 0


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise CanonicalProtocolError(f"metadata field {field!r} must be a non-empty string")
    return value


def _sha256(value: Any, field: str) -> str:
    result = _string(value, field)
    if _SHA256_RE.fullmatch(result) is None:
        raise CanonicalProtocolError(f"metadata field {field!r} must be a lowercase SHA-256")
    return result


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CanonicalProtocolError(f"metadata field {field!r} must be an object")
    return value


def _append_repeatable(argv: list[str], flag: str, values: Iterable[str]) -> None:
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{flag} values must be non-empty strings")
        argv.extend((flag, value))


def _selector_json(selector: Mapping[str, Any] | str) -> str:
    if isinstance(selector, str):
        if not selector:
            raise ValueError("conversation selectors must not be empty")
        return selector
    if not isinstance(selector, Mapping):
        raise TypeError("conversation selectors must be mappings or JSON strings")
    return json.dumps(selector, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def build_chunks_argv(
    db: str | os.PathLike[str],
    *,
    canonical_root: str | os.PathLike[str] | None = None,
    input_keys: Iterable[str] = (),
    channels: Iterable[str] = (),
    speakers: Iterable[str] = (),
    conversation_selectors: Iterable[Mapping[str, Any] | str] = (),
    all_acquisitions_conversation_ids: Iterable[str] = (),
    nonblank_text: bool = False,
    blank_text: bool = False,
    time_roles: Iterable[str] = (),
    limit: int = DEFAULT_PAGE_LIMIT,
    executable: str | os.PathLike[str] = DEFAULT_EXECUTABLE,
) -> list[str]:
    """Build the one-process, all-pages argv for a canonical chunk traversal."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if nonblank_text and blank_text:
        raise ValueError("nonblank_text and blank_text are mutually exclusive")

    executable_value = os.fspath(executable)
    db_value = os.fspath(db)
    if not executable_value:
        raise ValueError("executable must not be empty")
    if not db_value:
        raise ValueError("db must not be empty")

    argv = [executable_value, "query", "chunks", "--db", db_value]
    if canonical_root is not None:
        root_value = os.fspath(canonical_root)
        if not root_value:
            raise ValueError("canonical_root must not be empty")
        argv.extend(("--canonical-root", root_value))
    argv.extend(("--format", "jsonl", "--all-pages", "--limit", str(limit)))

    _append_repeatable(argv, "--input-key", input_keys)
    _append_repeatable(argv, "--channel", channels)
    _append_repeatable(argv, "--speaker", speakers)
    for selector in conversation_selectors:
        argv.extend(("--conversation-selector", _selector_json(selector)))
    _append_repeatable(
        argv,
        "--all-acquisitions-conversation-id",
        all_acquisitions_conversation_ids,
    )
    if nonblank_text:
        argv.append("--nonblank-text")
    if blank_text:
        argv.append("--blank-text")
    _append_repeatable(argv, "--time-role", time_roles)
    return argv


class CanonicalChunkStream:
    """Single-use incremental traversal of native canonical chunk rows.

    Accessing :attr:`provenance` starts the executable and reads only the first
    metadata record. Iterating then yields rows one at a time. Exhaustion is
    required to certify the subprocess exit code and terminal page framing.
    """

    def __init__(
        self,
        db: str | os.PathLike[str],
        *,
        canonical_root: str | os.PathLike[str] | None = None,
        input_keys: Iterable[str] = (),
        channels: Iterable[str] = (),
        speakers: Iterable[str] = (),
        conversation_selectors: Iterable[Mapping[str, Any] | str] = (),
        all_acquisitions_conversation_ids: Iterable[str] = (),
        nonblank_text: bool = False,
        blank_text: bool = False,
        time_roles: Iterable[str] = (),
        limit: int = DEFAULT_PAGE_LIMIT,
        executable: str | os.PathLike[str] = DEFAULT_EXECUTABLE,
    ) -> None:
        self.argv = build_chunks_argv(
            db,
            canonical_root=canonical_root,
            input_keys=input_keys,
            channels=channels,
            speakers=speakers,
            conversation_selectors=conversation_selectors,
            all_acquisitions_conversation_ids=all_acquisitions_conversation_ids,
            nonblank_text=nonblank_text,
            blank_text=blank_text,
            time_roles=time_roles,
            limit=limit,
            executable=executable,
        )
        self._limit = limit
        self._process: subprocess.Popen[str] | None = None
        self._stdout: TextIO | None = None
        self._stderr_file: TextIO | None = None
        self._provenance: CanonicalQueryProvenance | None = None
        self._first_metadata: dict[str, Any] | None = None
        self._page: _PageState | None = None
        self._seen_page_tokens: set[str] = set()
        self._seen_chunk_ids: set[str] = set()
        self._line_number = 0
        self._iteration_started = False
        self._finished = False
        self._closed = False

    @property
    def provenance(self) -> CanonicalQueryProvenance:
        """The first page's verified provenance and invariant query identity."""
        self._start()
        assert self._provenance is not None
        return self._provenance

    @property
    def first_metadata(self) -> dict[str, Any]:
        """A defensive copy of the first page's complete metadata object."""
        self._start()
        assert self._first_metadata is not None
        return copy.deepcopy(self._first_metadata)

    def __enter__(self: _StreamT) -> _StreamT:  # noqa: PYI019
        self._start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._iteration_started:
            raise CanonicalInputError("canonical chunk streams are single-use")
        self._iteration_started = True
        return self._iter_rows()

    def close(self) -> None:
        """Stop an unfinished traversal; completed traversals are already closed."""
        if not self._finished:
            self._abort()

    def _start(self) -> None:
        if self._process is not None or self._finished:
            return
        if self._closed:
            raise CanonicalInputError("canonical chunk stream is closed")
        # The file must stay open for the subprocess lifetime, so a lexical
        # context manager cannot own it here.
        self._stderr_file = tempfile.TemporaryFile(mode="w+t", encoding="utf-8")  # noqa: SIM115
        try:
            self._process = subprocess.Popen(
                self.argv,
                stdout=subprocess.PIPE,
                stderr=self._stderr_file,
                text=True,
                encoding="utf-8",
                errors="strict",
                shell=False,
            )
        except OSError as error:
            self._stderr_file.close()
            self._stderr_file = None
            self._closed = True
            raise CanonicalQueryError(f"could not start {self.argv[0]!r}: {error}") from error

        assert self._process.stdout is not None
        self._stdout = self._process.stdout
        try:
            record = self._read_record()
            if record is None:
                self._finish_process()
                raise CanonicalProtocolError("canonical query produced no metadata record")
            self._accept_metadata(record, first=True)
        except BaseException:
            self._abort()
            raise

    def _read_record(self) -> dict[str, Any] | None:
        assert self._stdout is not None
        try:
            line = self._stdout.readline()
        except UnicodeError as error:
            raise CanonicalProtocolError("canonical query output is not valid UTF-8") from error
        if line == "":
            return None
        self._line_number += 1
        if not line.endswith("\n"):
            raise CanonicalProtocolError(
                f"canonical query line {self._line_number} is not newline-terminated"
            )
        if not line.strip():
            raise CanonicalProtocolError(f"canonical query line {self._line_number} is blank")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise CanonicalProtocolError(
                f"canonical query line {self._line_number} is malformed JSON: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise CanonicalProtocolError(
                f"canonical query line {self._line_number} must be a JSON object"
            )
        return value

    def _parse_provenance(self, metadata: Mapping[str, Any]) -> CanonicalQueryProvenance:
        contract = _string(metadata.get("query_contract_version"), "query_contract_version")
        if contract != QUERY_CONTRACT_VERSION:
            raise CanonicalProtocolError(f"unsupported query contract version {contract!r}")

        projection = _string(metadata.get("projection_version"), "projection_version")
        if projection != CHUNK_PROJECTION_VERSION:
            raise CanonicalProtocolError(f"unexpected chunk projection version {projection!r}")

        provenance = _mapping(metadata.get("provenance"), "provenance")
        if provenance.get("verified_open") is not True:
            raise CanonicalProtocolError("canonical provenance must declare verified_open=true")
        query = _mapping(metadata.get("query"), "query")
        verb = _string(query.get("verb"), "query.verb")
        ordering = _string(query.get("ordering"), "query.ordering")
        if verb != CHUNK_QUERY_VERB:
            raise CanonicalProtocolError(f"unexpected canonical query verb {verb!r}")
        if ordering != CHUNK_ORDERING:
            raise CanonicalProtocolError(f"unexpected canonical query ordering {ordering!r}")

        return CanonicalQueryProvenance(
            query_contract_version=contract,
            projection_version=projection,
            canonical_contract_version=_string(
                provenance.get("canonical_contract_version"),
                "provenance.canonical_contract_version",
            ),
            sqlite_schema_version=_string(
                provenance.get("sqlite_schema_version"),
                "provenance.sqlite_schema_version",
            ),
            generation_id=_string(provenance.get("generation_id"), "provenance.generation_id"),
            database_sha256=_sha256(
                provenance.get("database_sha256"), "provenance.database_sha256"
            ),
            logical_digest_sha256=_sha256(
                provenance.get("logical_digest_sha256"),
                "provenance.logical_digest_sha256",
            ),
            verification_report_id=_string(
                provenance.get("verification_report_id"),
                "provenance.verification_report_id",
            ),
            verification_report_contract_version=_string(
                provenance.get("verification_report_contract_version"),
                "provenance.verification_report_contract_version",
            ),
            verification_report_verified_at=_string(
                provenance.get("verification_report_verified_at"),
                "provenance.verification_report_verified_at",
            ),
            verified_open=True,
            query_verb=verb,
            ordering=ordering,
            query_fingerprint_sha256=_sha256(
                query.get("query_fingerprint_sha256"),
                "query.query_fingerprint_sha256",
            ),
        )

    def _parse_page(self, metadata: Mapping[str, Any]) -> _PageState:
        pagination = _mapping(metadata.get("pagination"), "pagination")
        limit = pagination.get("limit")
        returned_count = pagination.get("returned_count")
        next_page_token = pagination.get("next_page_token")
        if type(limit) is not int or limit != self._limit:
            raise CanonicalProtocolError(
                f"pagination.limit must remain equal to requested limit {self._limit}"
            )
        if type(returned_count) is not int or not 0 <= returned_count <= limit:
            raise CanonicalProtocolError(
                "pagination.returned_count must be an integer between zero and limit"
            )
        if next_page_token is not None and (
            not isinstance(next_page_token, str) or not next_page_token
        ):
            raise CanonicalProtocolError(
                "pagination.next_page_token must be null or a non-empty string"
            )
        return _PageState(returned_count, next_page_token)

    def _accept_metadata(self, record: Mapping[str, Any], *, first: bool) -> None:
        if record.get("record_type") != "metadata":
            position = "first record" if first else f"line {self._line_number}"
            raise CanonicalProtocolError(f"canonical query {position} must be metadata")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            raise CanonicalProtocolError("metadata records must contain a metadata object")

        parsed = self._parse_provenance(metadata)
        if first:
            self._provenance = parsed
            self._first_metadata = copy.deepcopy(metadata)
        elif parsed != self._provenance:
            raise CanonicalProtocolError(
                "canonical generation, provenance, query, projection, ordering, or "
                "fingerprint changed across pages"
            )
        page = self._parse_page(metadata)
        if page.next_page_token is not None:
            if page.returned_count == 0:
                raise CanonicalProtocolError(
                    "nonterminal canonical pages must emit at least one row"
                )
            if page.next_page_token in self._seen_page_tokens:
                raise CanonicalProtocolError(
                    "canonical pagination repeated a next_page_token"
                )
            self._seen_page_tokens.add(page.next_page_token)
        self._page = page

    def _finish_page(self, *, expecting_another: bool) -> None:
        assert self._page is not None
        if self._page.rows_seen != self._page.returned_count:
            raise CanonicalProtocolError(
                "page row count does not match metadata pagination.returned_count"
            )
        has_next = self._page.next_page_token is not None
        if expecting_another and not has_next:
            raise CanonicalProtocolError("metadata appeared after a terminal page")
        if not expecting_another and has_next:
            raise CanonicalProtocolError("canonical query ended before the advertised next page")

    def _accept_row(self, record: Mapping[str, Any]) -> dict[str, Any]:
        assert self._page is not None
        if self._page.rows_seen >= self._page.returned_count:
            raise CanonicalProtocolError("page emitted more rows than metadata declared")
        row = record.get("row")
        if not isinstance(row, dict):
            raise CanonicalProtocolError("row records must contain a row object")
        if not isinstance(row.get("chunk_id"), str) or not row["chunk_id"]:
            raise CanonicalProtocolError("canonical chunk rows require a non-empty chunk_id")
        if not isinstance(row.get("text"), str):
            raise CanonicalProtocolError("canonical chunk rows require string text")
        chunk_id = row["chunk_id"]
        if chunk_id in self._seen_chunk_ids:
            raise CanonicalProtocolError(
                f"canonical query repeated chunk_id {chunk_id!r}"
            )
        self._seen_chunk_ids.add(chunk_id)
        self._page.rows_seen += 1
        return row

    def _iter_rows(self) -> Iterator[dict[str, Any]]:
        self._start()
        try:
            while True:
                record = self._read_record()
                if record is None:
                    self._finish_process()
                    self._finish_page(expecting_another=False)
                    self._finished = True
                    return
                record_type = record.get("record_type")
                if record_type == "metadata":
                    self._finish_page(expecting_another=True)
                    self._accept_metadata(record, first=False)
                elif record_type == "row":
                    yield self._accept_row(record)
                else:
                    raise CanonicalProtocolError(
                        f"canonical query line {self._line_number} has unknown record_type"
                    )
        except BaseException:
            self._abort()
            raise

    def _stderr(self) -> str:
        if self._stderr_file is None:
            return ""
        self._stderr_file.flush()
        self._stderr_file.seek(0)
        return self._stderr_file.read().strip()

    def _finish_process(self) -> None:
        assert self._process is not None
        return_code = self._process.wait()
        stderr = self._stderr()
        if self._stdout is not None:
            self._stdout.close()
            self._stdout = None
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        self._process = None
        if return_code != 0:
            detail = f": {stderr}" if stderr else ""
            raise CanonicalQueryError(f"canonical query exited with status {return_code}{detail}")

    def _abort(self) -> None:
        process = self._process
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if self._stdout is not None:
                self._stdout.close()
        if self._stderr_file is not None:
            self._stderr_file.close()
        self._process = None
        self._stdout = None
        self._stderr_file = None
        self._closed = True


def open_canonical_chunks(
    db: str | Path,
    **kwargs: Any,
) -> CanonicalChunkStream:
    """Construct a canonical chunk stream without any legacy fallback path."""
    return CanonicalChunkStream(db, **kwargs)
