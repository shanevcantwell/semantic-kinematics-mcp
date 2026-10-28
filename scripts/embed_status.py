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
    with checkpoint_lock(checkpoint, exclusive=False):
        reconcile_source_sidecar(
            checkpoint,
            source.provenance,
            write_if_missing=False,
            lock_held=True,
        )
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
                    cid = obj.get("chunk_id")
                    if cid is None:
                        continue
                    cid = str(cid)
                    if obj.get("_failed"):
                        failed.add(cid)
                    elif isinstance(obj.get("embedding"), list) and obj["embedding"]:
                        done.add(cid)

    extra_ids = (done | failed) - source_ids
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
        reconcile_source_sidecar(
            args.checkpoint, source.provenance, write_if_missing=False
        )
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
