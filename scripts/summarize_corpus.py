#!/usr/bin/env python3
"""Generate a markdown summary of embedding-store completion state.

The "stop hand-writing derived stats" fix: every number in the generated
report is *derived from ground truth* (the corpus JSONL + the checkpoint
JSONL + the checkpoint's ``.meta.json`` sidecar, if any) at run time — never
copied from a prior report or a Slack message. Shape adapted from
``llauncher/scripts/summarize_tests.py`` (OVERWRITE-IN-PLACE markdown,
do-not-hand-edit banner, Overview table + per-item detail sections).

Reuses ``scripts.embed_status.compute_status`` verbatim for the
done/failed/pending/total derivation (dedup-keep-last, exclude ``_failed``,
exclude zero-norm) rather than reimplementing that logic — see
``embed_status.py`` for the exact semantics. ``compute_status`` already reads
the checkpoint line-by-line (a generator loop over ``open(checkpoint)``), so
it does not load multi-gigabyte checkpoints into memory; the only full-file
read is the corpus itself (small — the embeddable-item list), which is
already required by ``compute_status``'s own ``total`` derivation and is
therefore not something this script can avoid without forking that function.

Run from anywhere:

.. code-block:: bash

    python scripts/summarize_corpus.py

Override the registry via env (``SK_CORPUS_<STORE>``, ``SK_CHECKPOINT_<STORE>``
with the store name upper-cased and ``-`` replaced by ``_``) or by editing the
``STORES`` list below directly — the paths are expected to move soon (the
sidecar-identity gap tracked as tvi#37), so this file is the one obvious place
to update them.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.embed_status import compute_status  # noqa: E402

OUTPUT_MARKDOWN_FILE = "docs/generated/CORPUS_STATS.md"


@dataclass
class StoreSpec:
    """One embedding store: a (corpus, checkpoint) pair to report on."""

    name: str
    corpus: str
    checkpoint: str
    text_field: str = "text"
    id_field: str = "chunk_id"


# ---------------------------------------------------------------------------
# STORES registry — the one obvious place to edit when a path moves.
#
# Paths are relative to the repo root and resolved robustly (symlinks are
# followed transparently by open(); no special-casing needed there).
# ---------------------------------------------------------------------------
STORES: list[StoreSpec] = [
    StoreSpec(
        name="embgemma-300m-768",
        corpus="../thought-vault-integration/output/vectors/chunks.jsonl",
        checkpoint="../thought-vault-integration/output/vectors/embed_checkpoint.jsonl",
    ),
    StoreSpec(
        name="nvembed-v2-4096",
        corpus="../thought-vault-integration/output/vectors/chunks.jsonl",
        checkpoint=(
            "../thought-vault-integration/output/thought-vault-integration-data/"
            "nv4096/corpus_4096.jsonl"
        ),
    ),
]


def _env_override(store: StoreSpec) -> StoreSpec:
    """Apply SK_CORPUS_<STORE> / SK_CHECKPOINT_<STORE> env overrides, if set."""
    key = store.name.upper().replace("-", "_")
    corpus = os.environ.get(f"SK_CORPUS_{key}", store.corpus)
    checkpoint = os.environ.get(f"SK_CHECKPOINT_{key}", store.checkpoint)
    return StoreSpec(store.name, corpus, checkpoint, store.text_field, store.id_field)


def _resolve(repo_root: Path, path: str) -> Path:
    """Resolve a registry path relative to the repo root, following symlinks."""
    p = Path(path)
    if not p.is_absolute():
        p = repo_root / p
    return p


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _read_meta(checkpoint_path: Path) -> tuple[str | None, int | None]:
    """Read ``<checkpoint>.meta.json`` -> (model_name, dimensions), or (None, None)."""
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if not meta_path.is_file():
        return None, None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None
    return data.get("model_name"), data.get("dimensions")


def _infer_dim_from_first_embedding(checkpoint_path: Path) -> int | None:
    """Fallback dim inference: length of the first non-failed embedding array.

    Used only when no ``.meta.json`` sidecar exists (e.g. the 768 store,
    tvi#37) — streams line-by-line and stops at the first hit, so this is
    safe to run against a large checkpoint.
    """
    if not checkpoint_path.is_file():
        return None
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("_failed"):
                continue
            emb = obj.get("embedding")
            if isinstance(emb, list) and emb:
                return len(emb)
    return None


def _raw_and_distinct_counts(checkpoint_path: Path) -> tuple[int, int]:
    """Return (raw_line_count, distinct_chunk_id_count) for the checkpoint.

    Streams line-by-line — safe against multi-gigabyte checkpoints.
    """
    raw = 0
    distinct: set[str] = set()
    if not checkpoint_path.is_file():
        return 0, 0
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid is not None:
                distinct.add(cid)
    return raw, len(distinct)


@dataclass
class StoreReport:
    spec: StoreSpec
    model_name: str
    dim: int | None
    done: int
    failed: int
    pending: int
    total: int
    checkpoint_exists: bool
    checkpoint_mtime_iso: str | None
    checkpoint_size_bytes: int
    raw_lines: int
    distinct_ids: int
    elapsed_seconds: float = field(default=0.0)

    @property
    def completion_pct(self) -> float:
        return (self.done / self.total * 100.0) if self.total else 0.0

    @property
    def duplicate_lines(self) -> int:
        return self.raw_lines - self.distinct_ids


def _build_report(repo_root: Path, spec: StoreSpec) -> StoreReport:
    corpus_path = _resolve(repo_root, spec.corpus)
    checkpoint_path = _resolve(repo_root, spec.checkpoint)

    start = time.monotonic()
    done, failed, pending, total = compute_status(
        str(corpus_path), str(checkpoint_path), spec.text_field, spec.id_field
    )
    raw_lines, distinct_ids = _raw_and_distinct_counts(checkpoint_path)
    elapsed = time.monotonic() - start

    model_name, dim = _read_meta(checkpoint_path)
    missing_sidecar = model_name is None
    if missing_sidecar:
        model_name = "(MISSING SIDECAR — tvi#37)"
        dim = _infer_dim_from_first_embedding(checkpoint_path)

    checkpoint_exists = checkpoint_path.is_file()
    mtime_iso = None
    size_bytes = 0
    if checkpoint_exists:
        st = checkpoint_path.stat()
        mtime_iso = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        size_bytes = st.st_size

    return StoreReport(
        spec=spec,
        model_name=model_name,
        dim=dim,
        done=done,
        failed=failed,
        pending=pending,
        total=total,
        checkpoint_exists=checkpoint_exists,
        checkpoint_mtime_iso=mtime_iso,
        checkpoint_size_bytes=size_bytes,
        raw_lines=raw_lines,
        distinct_ids=distinct_ids,
        elapsed_seconds=elapsed,
    )


def _render(reports: list[StoreReport]) -> str:
    lines: list[str] = [
        "# Corpus Stats",
        "",
        "> **Auto-generated by `scripts/summarize_corpus.py`** — do not hand-edit.",
        "> Regenerate:",
        ">",
        "> ```bash",
        "> python scripts/summarize_corpus.py",
        "> ```",
        ">",
        "> Every number below is derived from the corpus + checkpoint + sidecar",
        "> ground truth at generation time via `scripts/embed_status.compute_status`",
        "> — this file is a snapshot, not a source of truth in itself.",
        "",
        "## Overview",
        "",
        "| Store | Model | Dim | Done | Failed | Pending | Total | Completion% | Checkpoint mtime (UTC) |",
        "|-------|-------|-----|------|--------|---------|-------|-------------|------------------------|",
    ]
    for r in reports:
        dim_str = str(r.dim) if r.dim is not None else "?"
        mtime_str = r.checkpoint_mtime_iso or "(no checkpoint)"
        lines.append(
            f"| {r.spec.name} | {r.model_name} | {dim_str} | {r.done} | {r.failed} | "
            f"{r.pending} | {r.total} | {r.completion_pct:.2f}% | {mtime_str} |"
        )
    lines.append("")

    for r in reports:
        lines.append(f"## {r.spec.name}")
        lines.append("")
        lines.append(f"- **Corpus**: `{r.spec.corpus}`")
        lines.append(f"- **Checkpoint**: `{r.spec.checkpoint}`")
        lines.append(f"- **Model**: {r.model_name}")
        lines.append(f"- **Dimensions**: {r.dim if r.dim is not None else '?'}")
        lines.append(
            f"- **Checkpoint size**: {_human_size(r.checkpoint_size_bytes)} "
            f"({r.checkpoint_size_bytes:,} bytes)"
        )
        lines.append(f"- **Checkpoint mtime**: {r.checkpoint_mtime_iso or '(no checkpoint)'} (UTC)")
        lines.append(f"- **Raw checkpoint lines**: {r.raw_lines:,}")
        lines.append(f"- **Distinct chunk_ids**: {r.distinct_ids:,}")
        lines.append(
            f"- **Duplicate/retry lines** (raw − distinct): {r.duplicate_lines:,}"
        )
        lines.append(f"- **Done**: {r.done:,}")
        lines.append(f"- **Failed (residual, not superseded)**: {r.failed:,}")
        lines.append(f"- **Pending**: {r.pending:,}")
        lines.append(f"- **Total embeddable items in corpus**: {r.total:,}")
        lines.append(f"- **Completion**: {r.completion_pct:.2f}%")
        lines.append(f"- **Scan wall-time**: {r.elapsed_seconds:.2f}s")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs/generated/CORPUS_STATS.md from ground truth."
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_MARKDOWN_FILE,
        help="Output markdown path, relative to repo root unless absolute.",
    )
    args = parser.parse_args(argv)

    repo_root = Path(_REPO_ROOT)
    output_path = _resolve(repo_root, args.output)

    reports = []
    for spec in STORES:
        spec = _env_override(spec)
        print(f"[summarize_corpus] scanning store {spec.name!r}...", file=sys.stderr)
        report = _build_report(repo_root, spec)
        print(
            f"[summarize_corpus] {spec.name}: {report.done}/{report.total} done "
            f"({report.completion_pct:.2f}%), {report.elapsed_seconds:.2f}s",
            file=sys.stderr,
        )
        reports.append(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(reports), encoding="utf-8")
    store_names = ", ".join(r.spec.name for r in reports)
    print(f"Wrote {output_path} ({len(reports)} stores: {store_names})")


if __name__ == "__main__":
    main()
