#!/usr/bin/env python
"""Bulk-embed either a generic JSONL corpus or verified TVI canonical rows.

The checkpoint remains a pure model-keyed JSONL vector boundary.  Corpus
identity is recorded separately in an atomic ``<checkpoint>.source.json``
sidecar, and resume refuses a source generation/query mismatch.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

# Allow running as a plain script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.tvi_canonical_input import (
    DEFAULT_EXECUTABLE,
    CanonicalQueryProvenance,
    open_canonical_chunks,
)
from semantic_kinematics.embeddings import get_adapter
from semantic_kinematics.embeddings.bulk import BulkEmbedder

CONSUMER_NAME = "semantic-kinematics.embed_corpus"
CONSUMER_VERSION = "1.0"
SOURCE_SIDECAR_SUFFIX = ".source.json"


@dataclass(frozen=True)
class CorpusSource:
    """Ordered embeddable items plus their normalized source identity."""

    items: list[tuple[str, str]]
    provenance: dict[str, Any]


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return file identity and mutation-sensitive metadata for one read."""
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


def _read_jsonl_items_and_sha256(
    path: str, text_field: str, id_field: str
) -> tuple[list[tuple[str, str]], str]:
    """Parse and hash the same stable JSONL byte stream in one pass."""
    items: list[tuple[str, str]] = []
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as source:
            before = os.fstat(source.fileno())
            for line_index, raw_line in enumerate(source):
                digest.update(raw_line)
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = obj.get(text_field)
                if not text or not str(text).strip():
                    continue
                chunk_id = obj.get(id_field)
                if chunk_id is None:
                    chunk_id = f"line-{line_index}"
                items.append((str(text), str(chunk_id)))
            after = os.fstat(source.fileno())
        path_after = os.stat(path)
    except OSError as exc:
        raise ValueError(
            f"generic JSONL source {path!r} could not be read stably ({exc}); refusing"
        ) from exc

    if _stat_identity(before) != _stat_identity(after) or _stat_identity(
        after
    ) != _stat_identity(path_after):
        raise ValueError(
            f"generic JSONL source {path!r} changed while being read; refusing"
        )
    return items, digest.hexdigest()


def _read_items(path: str, text_field: str, id_field: str) -> list[tuple[str, str]]:
    """Read ``(text, chunk_id)`` items from one stable JSONL snapshot."""
    return _read_jsonl_items_and_sha256(path, text_field, id_field)[0]


def _normalize_input_keys(input_keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(sorted(input_keys))
    if not keys:
        raise ValueError("canonical mode requires at least one --input-key")
    if any(not key for key in keys):
        raise ValueError("--input-key values must be non-empty")
    if len(set(keys)) != len(keys):
        raise ValueError("--input-key values must not contain duplicates")
    return keys


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _canonical_provenance(
    provenance: CanonicalQueryProvenance,
    *,
    corpus_db: str,
    input_keys: tuple[str, ...],
) -> dict[str, Any]:
    block = provenance.as_dict()
    block.update(
        {
            "source_kind": "tvi_canonical_db",
            "database_path": os.path.abspath(corpus_db),
            "normalized_input_keys": list(input_keys),
            "normalized_filters": {
                "channels": [],
                "speakers": [],
                "conversations": [],
                "nonblank_text": None,
                "time_roles": [],
            },
            "consumer": {"name": CONSUMER_NAME, "version": CONSUMER_VERSION},
            "produced_at": _utc_now(),
        }
    )
    return block


def _read_canonical_items(
    corpus_db: str,
    *,
    canonical_root: str | None,
    input_keys: tuple[str, ...],
    executable: str,
) -> CorpusSource:
    """Read canonical rows in query-door order with JSONL-equivalent blanks."""
    items: list[tuple[str, str]] = []
    with open_canonical_chunks(
        corpus_db,
        canonical_root=canonical_root,
        input_keys=input_keys,
        executable=executable,
    ) as stream:
        provenance = stream.provenance
        for row in stream:
            text = row["text"]
            if not text or not text.strip():
                continue
            items.append((text, row["chunk_id"]))
    return CorpusSource(
        items,
        _canonical_provenance(
            provenance,
            corpus_db=corpus_db,
            input_keys=input_keys,
        ),
    )


def _read_jsonl_source(path: str, text_field: str, id_field: str) -> CorpusSource:
    resolved = os.path.abspath(path)
    items, sha256 = _read_jsonl_items_and_sha256(path, text_field, id_field)
    return CorpusSource(
        items,
        {
            "source_kind": "legacy_jsonl",
            "path": resolved,
            "sha256": sha256,
            "text_field": text_field,
            "id_field": id_field,
            "consumer": {"name": CONSUMER_NAME, "version": CONSUMER_VERSION},
            "produced_at": _utc_now(),
        },
    )


def load_corpus_source(
    *,
    corpus: str | None = None,
    corpus_db: str | None = None,
    canonical_root: str | None = None,
    input_keys: Iterable[str] = (),
    text_field: str = "text",
    id_field: str = "chunk_id",
    tvi_executable: str = DEFAULT_EXECUTABLE,
) -> CorpusSource:
    """Load exactly one explicit input mode; canonical failures never fallback."""
    if bool(corpus) == bool(corpus_db):
        raise ValueError("select exactly one source: generic JSONL corpus or --corpus-db")
    if corpus_db:
        normalized_keys = _normalize_input_keys(input_keys)
        return _read_canonical_items(
            corpus_db,
            canonical_root=canonical_root,
            input_keys=normalized_keys,
            executable=tvi_executable,
        )
    if canonical_root or tuple(input_keys):
        raise ValueError("--canonical-root/--input-key are valid only with --corpus-db")
    assert corpus is not None
    return _read_jsonl_source(corpus, text_field, id_field)


def _source_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    kind = provenance.get("source_kind")
    if kind == "tvi_canonical_db":
        query = provenance.get("query")
        if not isinstance(query, dict):
            raise ValueError("canonical source provenance has no query identity")
        return {
            "source_kind": kind,
            "generation_id": provenance.get("generation_id"),
            "query_fingerprint_sha256": query.get("query_fingerprint_sha256"),
        }
    if kind == "legacy_jsonl":
        return {
            "source_kind": kind,
            "path": provenance.get("path"),
            "sha256": provenance.get("sha256"),
            "text_field": provenance.get("text_field"),
            "id_field": provenance.get("id_field"),
        }
    raise ValueError(f"unsupported checkpoint source kind: {kind!r}")


def source_sidecar_path(checkpoint: str) -> str:
    return checkpoint + SOURCE_SIDECAR_SUFFIX


def checkpoint_lock_path(checkpoint: str) -> str:
    return source_sidecar_path(checkpoint) + ".lock"


@contextmanager
def checkpoint_lock(checkpoint: str, *, exclusive: bool) -> Iterator[None]:
    """Coordinate source reconciliation and checkpoint access across processes."""
    path = checkpoint_lock_path(checkpoint)
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "a+b") as lock_file:
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), operation)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _reconcile_source_sidecar_unlocked(
    checkpoint: str,
    provenance: dict[str, Any],
    *,
    write_if_missing: bool,
) -> dict[str, Any]:
    path = source_sidecar_path(checkpoint)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as source:
                existing = json.load(source)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"checkpoint source provenance {path!r} is unreadable ({exc}); "
                "refusing to resume"
            ) from exc
        existing_identity = _source_identity(existing)
        current_identity = _source_identity(provenance)
        if existing_identity != current_identity:
            raise ValueError(
                f"checkpoint {checkpoint!r} source mismatch: "
                f"existing={existing_identity}, current={current_identity}; "
                "refusing to mix generations or queries"
            )
        return existing

    if os.path.exists(checkpoint) and os.path.getsize(checkpoint) > 0:
        raise ValueError(
            f"checkpoint {checkpoint!r} is populated but has no source provenance "
            f"sidecar {path!r}; refusing unsafe legacy adoption"
        )
    if not write_if_missing:
        return provenance

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=os.path.basename(path) + ".", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(provenance, output, ensure_ascii=False, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return provenance


def reconcile_source_sidecar(
    checkpoint: str,
    provenance: dict[str, Any],
    *,
    write_if_missing: bool = True,
    lock_held: bool = False,
) -> dict[str, Any]:
    """Create/validate provenance while refusing unproven populated checkpoints."""
    if lock_held:
        return _reconcile_source_sidecar_unlocked(
            checkpoint, provenance, write_if_missing=write_if_missing
        )
    with checkpoint_lock(checkpoint, exclusive=True):
        return _reconcile_source_sidecar_unlocked(
            checkpoint, provenance, write_if_missing=write_if_missing
        )


def source_summary(provenance: dict[str, Any]) -> str:
    if provenance["source_kind"] == "tvi_canonical_db":
        keys = ",".join(provenance["normalized_input_keys"])
        return (
            f"source_kind=tvi_canonical_db generation_id={provenance['generation_id']} "
            f"logical_digest_sha256={provenance['logical_digest_sha256']} "
            f"input_keys={keys}"
        )
    return (
        f"source_kind=legacy_jsonl path={provenance['path']} "
        f"sha256={provenance['sha256']}"
    )


def _make_adapter(backend: str, model: str | None, base_url: str | None):
    """Build an embedding adapter with backend-appropriate constructor kwargs."""
    if backend == "lmstudio":
        return get_adapter(backend, model_name=model, base_url=base_url)
    if backend == "nv_embed":
        return get_adapter(backend, unload_after_use=False)
    return get_adapter(backend)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Bulk-embed generic JSONL or verified TVI canonical rows."
    )
    parser.add_argument("corpus", nargs="?", help="generic/non-TVI JSONL corpus")
    parser.add_argument("--corpus-db", help="verified TVI canonical corpus.db")
    parser.add_argument("--canonical-root", help="optional trusted canonical root")
    parser.add_argument(
        "--input-key",
        action="append",
        dest="input_keys",
        default=[],
        help="canonical input key (repeatable; required with --corpus-db)",
    )
    parser.add_argument(
        "--tvi-executable",
        default=os.environ.get("TVI_CANONICAL_CORPUS_EXECUTABLE", DEFAULT_EXECUTABLE),
        help="installed tvi-canonical-corpus executable",
    )
    parser.add_argument("--checkpoint", required=True, help="checkpoint JSONL path")
    parser.add_argument("--backend", default="lmstudio", help="embedding backend")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EMBEDDING_SERVER_URL"),
        help="lmstudio endpoint (arg -> EMBEDDING_SERVER_URL -> hard fail)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EMBEDDING_MODEL"),
        help="lmstudio model id (arg -> EMBEDDING_MODEL -> hard fail)",
    )
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="chunk_id")
    parser.add_argument("--max-tokens-per-request", type=int, default=3000)
    parser.add_argument("--max-tokens-per-chunk", type=int, default=1500)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if bool(args.corpus) == bool(args.corpus_db):
        parser.error("provide exactly one of generic CORPUS or --corpus-db")
    if args.corpus and (args.canonical_root or args.input_keys):
        parser.error("--canonical-root/--input-key require --corpus-db")
    if args.corpus_db:
        try:
            args.input_keys = list(_normalize_input_keys(args.input_keys))
        except ValueError as exc:
            parser.error(str(exc))
    if args.backend == "lmstudio" and not args.model:
        parser.error("--model is required for --backend lmstudio (or set EMBEDDING_MODEL)")
    if args.backend == "lmstudio" and not args.base_url:
        parser.error(
            "--base-url is required for --backend lmstudio "
            "(or set EMBEDDING_SERVER_URL)"
        )

    source = load_corpus_source(
        corpus=args.corpus,
        corpus_db=args.corpus_db,
        canonical_root=args.canonical_root,
        input_keys=args.input_keys,
        text_field=args.text_field,
        id_field=args.id_field,
        tvi_executable=args.tvi_executable,
    )
    with checkpoint_lock(args.checkpoint, exclusive=True):
        reconcile_source_sidecar(
            args.checkpoint, source.provenance, lock_held=True
        )
        print(f"[embed_corpus] {source_summary(source.provenance)}", file=sys.stderr)
        print(f"[embed_corpus] loaded {len(source.items)} items", file=sys.stderr)

        adapter = _make_adapter(args.backend, args.model, args.base_url)
        embedder = BulkEmbedder(
            adapter,
            max_tokens_per_request=args.max_tokens_per_request,
            max_tokens_per_chunk=args.max_tokens_per_chunk,
            checkpoint_path=args.checkpoint,
        )

        start = time.time()
        results = embedder.embed_corpus(source.items)
        elapsed = time.time() - start
    print(
        f"[embed_corpus] embedded {len(results)} / {len(source.items)} items "
        f"in {elapsed:.1f}s -> {args.checkpoint}"
    )


if __name__ == "__main__":
    main()
