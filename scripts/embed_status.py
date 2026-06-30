#!/usr/bin/env python
"""Report bulk-embed progress for a (corpus, checkpoint) pair.

Prints four space-separated integers: ``done failed pending total`` where

  total   = embeddable items in the corpus (blank-text lines skipped, matching
            embed_corpus._read_items)
  done    = distinct chunk_ids with a valid (non-failed, non-empty) embedding
  failed  = distinct chunk_ids present ONLY as ``_failed`` markers (not done)
  pending = total - done   (items still needing a successful embedding)

This is the truthful completion signal the run wrapper needs: ``embed_corpus``
exits 0 even when items are marked ``_failed``, and a raw ``wc -l`` of the
checkpoint counts failures as progress. ``done == total`` is the only real
"fully embedded" condition; ``pending > 0`` with ``failed > 0`` and no new
``done`` across passes means persistent per-item failures, not transient crashes.
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.embed_corpus import _read_items  # noqa: E402


def compute_status(corpus, checkpoint, text_field="text", id_field="chunk_id"):
    """Return (done, failed, pending, total) for the run."""
    items = _read_items(corpus, text_field, id_field)
    total = len(items)

    done = set()
    failed = set()
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
                if obj.get("_failed"):
                    failed.add(cid)
                elif isinstance(obj.get("embedding"), list) and obj["embedding"]:
                    done.add(cid)

    failed -= done  # a later success supersedes an earlier _failed marker
    pending = total - len(done)
    return len(done), len(failed), pending, total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report bulk-embed progress.")
    parser.add_argument("corpus")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--id-field", default="chunk_id")
    args = parser.parse_args(argv)

    done, failed, pending, total = compute_status(
        args.corpus, args.checkpoint, args.text_field, args.id_field
    )
    print(f"{done} {failed} {pending} {total}")


if __name__ == "__main__":
    main()
