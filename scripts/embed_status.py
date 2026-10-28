#!/usr/bin/env python
"""Report truthful bulk-embed progress for an explicit corpus source.

The input may be generic JSONL or a verified TVI canonical DB query.  The
checkpoint itself remains JSONL because it is the model-keyed vector boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.embed_corpus import (
    DEFAULT_EXECUTABLE,
    checkpoint_lock,
    load_corpus_source,
    reconcile_source_sidecar,
    source_summary,
)
from semantic_kinematics.embeddings.bulk import is_valid_embedding_record


def _checkpoint_is_populated(checkpoint: str) -> bool:
    """Return whether a checkpoint contains any non-whitespace record data."""
    if not os.path.isfile(checkpoint):
        return False
    with open(checkpoint, "r", encoding="utf-8") as checkpoint_file:
        return any(line.strip() for line in checkpoint_file)


def load_checkpoint_metadata(checkpoint: str) -> tuple[str | None, int | None]:
    """Return authoritative model/dimensions, failing closed when populated.

    An absent or whitespace-only checkpoint has no completed records, so it may
    legitimately have no metadata. Once any record data exists, the bulk
    sidecar contract requires a readable JSON object with a non-empty model name
    and a positive integer dimension.
    """
    if not _checkpoint_is_populated(checkpoint):
        return None, None

    meta_path = checkpoint + ".meta.json"
    try:
        with open(meta_path, "r", encoding="utf-8") as meta_file:
            metadata = json.load(meta_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"populated checkpoint metadata {meta_path!r} is missing or unreadable: {exc}"
        ) from exc

    if not isinstance(metadata, dict):
        raise ValueError(  # noqa: TRY004 -- public status contract uses ValueError.
            f"populated checkpoint metadata {meta_path!r} must be a JSON object"
        )
    model_name = metadata.get("model_name")
    dimensions = metadata.get("dimensions")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"populated checkpoint metadata {meta_path!r} has invalid model_name")
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise ValueError(f"populated checkpoint metadata {meta_path!r} has invalid dimensions")
    return model_name, dimensions


def compute_status(
    corpus: str | None,
    checkpoint: str,
    text_field: str = "text",
    id_field: str = "chunk_id",
    *,
    corpus_db: str | None = None,
    canonical_root: str | None = None,
    input_keys: Iterable[str] = (),
    tvi_executable: str = DEFAULT_EXECUTABLE,
) -> tuple[int, int, int, int]:
    """Return ``(done, failed, pending, total)`` for either source mode."""
    source = load_corpus_source(
        corpus=corpus,
        corpus_db=corpus_db,
        canonical_root=canonical_root,
        input_keys=input_keys,
        text_field=text_field,
        id_field=id_field,
        tvi_executable=tvi_executable,
    )
    source_ids_list = [chunk_id for _text, chunk_id in source.items]
    source_ids = set(source_ids_list)
    if len(source_ids) != len(source_ids_list):
        seen: set[str] = set()
        duplicates: set[str] = set()
        for chunk_id in source_ids_list:
            if chunk_id in seen:
                duplicates.add(chunk_id)
            seen.add(chunk_id)
        raise ValueError(
            "source contains duplicate chunk_id values: "
            + ", ".join(repr(value) for value in sorted(duplicates))
        )
    total = len(source_ids)

    done: set[str] = set()
    failed: set[str] = set()
    checkpoint_ids: set[str] = set()
    with checkpoint_lock(checkpoint, exclusive=False):
        reconcile_source_sidecar(
            checkpoint,
            source.provenance,
            write_if_missing=False,
            lock_held=True,
        )
        _model_name, expected_dimensions = load_checkpoint_metadata(checkpoint)
        if os.path.exists(checkpoint):
            with open(checkpoint, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    cid = obj.get("chunk_id")
                    if cid is None:
                        continue
                    cid = str(cid)
                    checkpoint_ids.add(cid)
                    if cid not in source_ids:
                        continue
                    if expected_dimensions is not None and is_valid_embedding_record(
                        obj, expected_dimensions
                    ):
                        done.add(cid)
                    elif obj.get("_failed"):
                        failed.add(cid)

    extra_ids = checkpoint_ids - source_ids
    if extra_ids:
        raise ValueError(
            "checkpoint contains chunk_id values absent from the current source: "
            + ", ".join(repr(value) for value in sorted(extra_ids))
        )
    failed -= done  # a later success supersedes an earlier _failed marker
    pending = len(source_ids - done)
    return len(done), len(failed), pending, total


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="chunk_id")
    parser.add_argument(
        "--source-summary",
        action="store_true",
        help="print source generation/digest (or generic path/hash) and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.corpus) == bool(args.corpus_db):
        parser.error("provide exactly one of generic CORPUS or --corpus-db")
    if args.corpus and (args.canonical_root or args.input_keys):
        parser.error("--canonical-root/--input-key require --corpus-db")

    # Summary mode deliberately uses the same loader and sidecar check as status,
    # so the wrapper cannot log one query identity and count another.
    if args.source_summary:
        source = load_corpus_source(
            corpus=args.corpus,
            corpus_db=args.corpus_db,
            canonical_root=args.canonical_root,
            input_keys=args.input_keys,
            text_field=args.text_field,
            id_field=args.id_field,
            tvi_executable=args.tvi_executable,
        )
        reconcile_source_sidecar(args.checkpoint, source.provenance, write_if_missing=False)
        print(source_summary(source.provenance))
        return

    try:
        done, failed, pending, total = compute_status(
            args.corpus,
            args.checkpoint,
            args.text_field,
            args.id_field,
            corpus_db=args.corpus_db,
            canonical_root=args.canonical_root,
            input_keys=args.input_keys,
            tvi_executable=args.tvi_executable,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{done} {failed} {pending} {total}")


if __name__ == "__main__":
    main()
