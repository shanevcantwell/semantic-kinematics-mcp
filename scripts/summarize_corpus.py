#!/usr/bin/env python3
"""Generate a markdown summary of embedding-store completion state.

TVI text totals and chunk IDs come only from the verified canonical query door.
Checkpoint JSONL remains the derived vector/status boundary: it is scanned for
valid vectors, failures, retries, dimensions, and authoritative model identity
when a metadata sidecar exists, but never used as corpus text authority.
Verified-open/query errors propagate
without a legacy JSONL fallback.

Run from anywhere:

.. code-block:: bash

    python scripts/summarize_corpus.py

Production store locations may be overridden with
``SK_CORPUS_DB_<STORE>``, ``SK_CHECKPOINT_<STORE>``,
``SK_CANONICAL_ROOT_<STORE>``, and ``SK_TVI_EXECUTABLE_<STORE>``.  Store names
are upper-cased with ``-`` replaced by ``_``.  The shared TVI environment
variables remain the defaults for the canonical DB/root/executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.embed_corpus import (
    DEFAULT_EXECUTABLE,
    load_corpus_source,
    source_sidecar_path,
)
from semantic_kinematics.embeddings.bulk import is_valid_embedding_record

OUTPUT_MARKDOWN_FILE = "docs/generated/CORPUS_STATS.md"
DEFAULT_CORPUS_DB = os.environ.get(
    "THOUGHT_VAULT_CANONICAL_DB",
    "/srv/dev/shanevcantwell/thought-vault-integration/output/canonical/corpus.db",
)
DEFAULT_CANONICAL_ROOT = os.environ.get("THOUGHT_VAULT_CANONICAL_ROOT")
DEFAULT_TVI_EXECUTABLE = os.environ.get("TVI_CANONICAL_CORPUS_EXECUTABLE", DEFAULT_EXECUTABLE)
DEFAULT_INPUT_KEYS = ("base",)


@dataclass(frozen=True)
class StoreSpec:
    """One text/ID source and its derived vector checkpoint.

    Production TVI stores use ``corpus_db``.  ``corpus`` remains available only
    for explicit generic/non-TVI fixtures and callers; the production registry
    below contains no JSONL corpus source.
    """

    name: str
    checkpoint: str
    corpus_db: str | None = None
    corpus: str | None = None
    canonical_root: str | None = None
    input_keys: tuple[str, ...] = ()
    tvi_executable: str = DEFAULT_TVI_EXECUTABLE
    text_field: str = "text"
    id_field: str = "chunk_id"

    def __post_init__(self) -> None:
        if bool(self.corpus_db) == bool(self.corpus):
            raise ValueError("StoreSpec requires exactly one of corpus_db or corpus")
        if self.corpus and (self.canonical_root or self.input_keys):
            raise ValueError("generic StoreSpec cannot declare canonical scope")


# TVI text/ID authority is shared across stores; only the vector checkpoint
# differs. Checkpoint reads remain vector-status reads; model identity is
# authoritative only when the checkpoint metadata sidecar is present.
STORES: list[StoreSpec] = [
    StoreSpec(
        name="embgemma-300m-768",
        corpus_db=DEFAULT_CORPUS_DB,
        canonical_root=DEFAULT_CANONICAL_ROOT,
        input_keys=DEFAULT_INPUT_KEYS,
        checkpoint=("../thought-vault-integration/output/vectors/embed_checkpoint.jsonl"),
    ),
    StoreSpec(
        name="nvembed-v2-4096",
        corpus_db=DEFAULT_CORPUS_DB,
        canonical_root=DEFAULT_CANONICAL_ROOT,
        input_keys=DEFAULT_INPUT_KEYS,
        checkpoint=(
            "../thought-vault-integration/output/thought-vault-integration-data/"
            "nv4096/corpus_4096.jsonl"
        ),
    ),
]


def _env_override(store: StoreSpec) -> StoreSpec:
    """Apply canonical-source/checkpoint overrides for a production store."""
    key = store.name.upper().replace("-", "_")
    checkpoint = os.environ.get(f"SK_CHECKPOINT_{key}", store.checkpoint)
    if store.corpus_db is None:
        return replace(store, checkpoint=checkpoint)
    corpus_db = os.environ.get(f"SK_CORPUS_DB_{key}", store.corpus_db)
    canonical_root = os.environ.get(f"SK_CANONICAL_ROOT_{key}", store.canonical_root)
    executable = os.environ.get(f"SK_TVI_EXECUTABLE_{key}", store.tvi_executable)
    return replace(
        store,
        checkpoint=checkpoint,
        corpus_db=corpus_db,
        canonical_root=canonical_root,
        tvi_executable=executable,
    )


def _resolve(repo_root: Path, path: str) -> Path:
    """Resolve a registry path relative to the repo root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    return candidate


def _resolved_optional(repo_root: Path, path: str | None) -> str | None:
    return str(_resolve(repo_root, path)) if path is not None else None


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def _read_metadata_sidecar(checkpoint_path: Path) -> tuple[str, int | None, str]:
    """Read model metadata without reopening the already-bound checkpoint."""
    meta_path = Path(str(checkpoint_path) + ".meta.json")
    if not meta_path.is_file():
        return (
            "(MISSING SIDECAR — tvi#37)",
            None,
            "legacy checkpoint: dimension inferred in snapshot; model identity not authoritative",
        )
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint metadata {meta_path!s} is unreadable: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError(  # noqa: TRY004 -- report contract consistently uses ValueError.
            f"checkpoint metadata {meta_path!s} must be a JSON object"
        )
    model_name = metadata.get("model_name")
    dimensions = metadata.get("dimensions")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError(f"checkpoint metadata {meta_path!s} has invalid model_name")
    if not isinstance(dimensions, int) or isinstance(dimensions, bool) or dimensions <= 0:
        raise ValueError(f"checkpoint metadata {meta_path!s} has invalid dimensions")
    return model_name, dimensions, "authoritative .meta.json"


def _report_source_identity(provenance: dict[str, Any]) -> dict[str, Any]:
    """Return the source fields that bind a checkpoint without mutating it."""
    source_kind = provenance.get("source_kind")
    if source_kind == "tvi_canonical_db":
        query = provenance.get("query")
        if not isinstance(query, dict):
            raise ValueError("canonical source provenance has no query identity")
        return {
            "source_kind": source_kind,
            "generation_id": provenance.get("generation_id"),
            "query_fingerprint_sha256": query.get("query_fingerprint_sha256"),
        }
    if source_kind == "legacy_jsonl":
        return {
            "source_kind": source_kind,
            "path": provenance.get("path"),
            "sha256": provenance.get("sha256"),
            "text_field": provenance.get("text_field"),
            "id_field": provenance.get("id_field"),
        }
    raise ValueError(f"unsupported checkpoint source kind: {source_kind!r}")


def _checkpoint_source_binding(checkpoint_path: Path, source_provenance: dict[str, Any]) -> str:
    """Validate an existing source sidecar, or label a legacy checkpoint unbound."""
    sidecar = Path(source_sidecar_path(str(checkpoint_path)))
    if not sidecar.is_file():
        return "legacy-unbound: no .source.json; no generation/query production claim"
    try:
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"checkpoint source provenance {sidecar!s} is unreadable: {exc}") from exc
    if _report_source_identity(existing) != _report_source_identity(source_provenance):
        raise ValueError(
            f"checkpoint {checkpoint_path!s} source identity does not match the reported source"
        )
    return "validated .source.json matches the reported source identity"


def _stat_identity(stat_result: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
        stat_result.st_ctime_ns,
    )


@dataclass(frozen=True)
class CheckpointSnapshot:
    model_name: str
    dimensions: int | None
    metadata_status: str
    source_binding: str
    source_bound: bool
    checkpoint_sha256: str | None
    done: int
    failed: int
    pending: int
    raw_lines: int
    distinct_ids: int
    stat_result: os.stat_result | None


def _scan_checkpoint_snapshot(
    checkpoint_path: Path,
    source_ids: set[str],
    source_provenance: dict[str, Any],
) -> CheckpointSnapshot:
    """Acquire metadata, inferred dimension, digest, and records from one snapshot.

    The checkpoint descriptor is opened before sidecar acquisition. Every record
    and any legacy dimension inference comes from that descriptor. ``fstat``
    before/after detects in-place mutation, while final path identity detects an
    atomic replacement even when size/mtime are preserved.
    """
    if not checkpoint_path.is_file():
        return CheckpointSnapshot(
            model_name="(MISSING SIDECAR — tvi#37)",
            dimensions=None,
            metadata_status="checkpoint absent or empty",
            source_binding="checkpoint absent; no source binding",
            source_bound=False,
            checkpoint_sha256=None,
            done=0,
            failed=0,
            pending=len(source_ids),
            raw_lines=0,
            distinct_ids=0,
            stat_result=None,
        )

    done: set[str] = set()
    failed: set[str] = set()
    checkpoint_ids: set[str] = set()
    raw_lines = 0
    digest = hashlib.sha256()

    with open(checkpoint_path, "rb") as checkpoint_file:
        stat_before = os.fstat(checkpoint_file.fileno())
        model_name, expected_dimensions, metadata_status = _read_metadata_sidecar(checkpoint_path)
        source_binding = _checkpoint_source_binding(checkpoint_path, source_provenance)
        source_bound = source_binding.startswith("validated ")

        for raw_line in checkpoint_file:
            digest.update(raw_line)
            line = raw_line.strip()
            if not line:
                continue
            raw_lines += 1
            try:
                obj = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(obj, dict) or obj.get("chunk_id") is None:
                continue
            embedding = obj.get("embedding")
            if expected_dimensions is None and isinstance(embedding, list) and len(embedding) > 0:
                expected_dimensions = len(embedding)
            chunk_id = str(obj["chunk_id"])
            checkpoint_ids.add(chunk_id)
            if chunk_id not in source_ids:
                continue
            if expected_dimensions is not None and is_valid_embedding_record(
                obj, expected_dimensions
            ):
                done.add(chunk_id)
            elif obj.get("_failed"):
                failed.add(chunk_id)

        stat_after = os.fstat(checkpoint_file.fileno())

    try:
        path_stat = checkpoint_path.stat()
    except FileNotFoundError as exc:
        raise ValueError(
            f"checkpoint {checkpoint_path!s} changed during report scan; retry"
        ) from exc
    identities = {
        _stat_identity(stat_before),
        _stat_identity(stat_after),
        _stat_identity(path_stat),
    }
    if len(identities) != 1:
        raise ValueError(f"checkpoint {checkpoint_path!s} changed during report scan; retry")

    extra_ids = checkpoint_ids - source_ids
    if extra_ids:
        raise ValueError(
            "checkpoint contains chunk_id values absent from the current source: "
            + ", ".join(repr(value) for value in sorted(extra_ids))
        )
    failed -= done
    if stat_before.st_size == 0:
        expected_dimensions = None
        metadata_status = "checkpoint absent or empty"
    return CheckpointSnapshot(
        model_name=model_name,
        dimensions=expected_dimensions,
        metadata_status=metadata_status,
        source_binding=source_binding,
        source_bound=source_bound,
        checkpoint_sha256=digest.hexdigest(),
        done=len(done),
        failed=len(failed),
        pending=len(source_ids - done),
        raw_lines=raw_lines,
        distinct_ids=len(checkpoint_ids),
        stat_result=stat_before,
    )


@dataclass
class StoreReport:
    spec: StoreSpec
    source_provenance: dict[str, Any]
    checkpoint_source_binding: str
    checkpoint_metadata_status: str
    checkpoint_source_bound: bool
    checkpoint_sha256: str | None
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
    def coverage_pct(self) -> float:
        return (self.done / self.total * 100.0) if self.total else 0.0

    @property
    def completion_pct(self) -> float:
        """Compatibility alias; rendering calls this completion only when bound."""
        return self.coverage_pct

    @property
    def duplicate_lines(self) -> int:
        return self.raw_lines - self.distinct_ids

    @property
    def generation_id(self) -> str | None:
        value = self.source_provenance.get("generation_id")
        return value if isinstance(value, str) else None


def _load_source(repo_root: Path, spec: StoreSpec):
    corpus = _resolved_optional(repo_root, spec.corpus)
    corpus_db = _resolved_optional(repo_root, spec.corpus_db)
    canonical_root = _resolved_optional(repo_root, spec.canonical_root)
    return load_corpus_source(
        corpus=corpus,
        corpus_db=corpus_db,
        canonical_root=canonical_root,
        input_keys=spec.input_keys if corpus_db is not None else (),
        text_field=spec.text_field,
        id_field=spec.id_field,
        tvi_executable=spec.tvi_executable,
    )


def _build_report(repo_root: Path, spec: StoreSpec) -> StoreReport:
    checkpoint_path = _resolve(repo_root, spec.checkpoint)

    start = time.monotonic()
    # Canonical acquisition completes before any checkpoint read. A verified-open
    # or query failure therefore fails closed at the text/ID boundary.
    source = _load_source(repo_root, spec)
    source_ids_list = [chunk_id for _text, chunk_id in source.items]
    source_ids = set(source_ids_list)
    if len(source_ids) != len(source_ids_list):
        raise ValueError("source contains duplicate chunk_id values")
    total = len(source_ids)

    snapshot = _scan_checkpoint_snapshot(checkpoint_path, source_ids, source.provenance)
    elapsed = time.monotonic() - start

    checkpoint_exists = snapshot.stat_result is not None
    mtime_iso = None
    size_bytes = 0
    if snapshot.stat_result is not None:
        mtime_iso = datetime.fromtimestamp(
            snapshot.stat_result.st_mtime, tz=timezone.utc
        ).isoformat()
        size_bytes = snapshot.stat_result.st_size

    return StoreReport(
        spec=spec,
        source_provenance=source.provenance,
        checkpoint_source_binding=snapshot.source_binding,
        checkpoint_metadata_status=snapshot.metadata_status,
        checkpoint_source_bound=snapshot.source_bound,
        checkpoint_sha256=snapshot.checkpoint_sha256,
        model_name=snapshot.model_name,
        dim=snapshot.dimensions,
        done=snapshot.done,
        failed=snapshot.failed,
        pending=snapshot.pending,
        total=total,
        checkpoint_exists=checkpoint_exists,
        checkpoint_mtime_iso=mtime_iso,
        checkpoint_size_bytes=size_bytes,
        raw_lines=snapshot.raw_lines,
        distinct_ids=snapshot.distinct_ids,
        elapsed_seconds=elapsed,
    )


def _render_source(lines: list[str], report: StoreReport) -> None:
    provenance = report.source_provenance
    source_kind = provenance.get("source_kind")
    if source_kind == "tvi_canonical_db":
        query = provenance["query"]
        lines.append("- **Text/ID source**: verified TVI canonical query")
        lines.append(f"- **Canonical DB**: `{provenance['database_path']}`")
        lines.append(f"- **Query executable**: `{report.spec.tvi_executable}`")
        lines.append(
            f"- **Trusted canonical root**: `{report.spec.canonical_root or '(not supplied)'}`"
        )
        lines.append(
            "- **Canonical input keys**: "
            + ", ".join(f"`{key}`" for key in provenance["normalized_input_keys"])
        )
        lines.append(f"- **Canonical generation ID**: `{provenance['generation_id']}`")
        lines.append(f"- **Database SHA-256**: `{provenance['database_sha256']}`")
        lines.append(f"- **Logical digest SHA-256**: `{provenance['logical_digest_sha256']}`")
        lines.append(
            "- **Verified query**: "
            f"`{query['verb']}` / `{query['ordering']}` / "
            f"fingerprint `{query['query_fingerprint_sha256']}`"
        )
        lines.append(f"- **Verified open**: `{str(provenance['verified_open']).lower()}`")
    else:
        lines.append(f"- **Text/ID source**: explicit generic JSONL `{provenance['path']}`")
        lines.append(f"- **Generic source SHA-256**: `{provenance['sha256']}`")


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
        "> TVI text totals and IDs come from a verified canonical query. Vector",
        "> ID coverage comes from checkpoint snapshots; completion is claimed only",
        "> when a matching source sidecar binds the checkpoint. This file is a snapshot.",
        "",
        "## Overview",
        "",
        "| Store | Queried canonical generation | Checkpoint generation binding | Model | Dim | Covered IDs | Failed | Uncovered | Queried total | ID coverage | Checkpoint mtime (UTC) |",
        "|-------|------------------------------|-------------------------------|-------|-----|------------:|-------:|----------:|--------------:|------------:|------------------------|",
    ]
    for report in reports:
        dim_str = str(report.dim) if report.dim is not None else "?"
        mtime_str = report.checkpoint_mtime_iso or "(no checkpoint)"
        generation = report.generation_id or "generic"
        binding = "validated" if report.checkpoint_source_bound else "unbound"
        lines.append(
            f"| {report.spec.name} | {generation} | {binding} | {report.model_name} | "
            f"{dim_str} | {report.done} | {report.failed} | {report.pending} | "
            f"{report.total} | {report.coverage_pct:.2f}% | {mtime_str} |"
        )
    lines.append("")

    for report in reports:
        lines.append(f"## {report.spec.name}")
        lines.append("")
        _render_source(lines, report)
        lines.append(f"- **Checkpoint (vector state)**: `{report.spec.checkpoint}`")
        lines.append(f"- **Checkpoint source binding**: {report.checkpoint_source_binding}")
        lines.append(
            "- **Checkpoint generation binding**: "
            + ("validated" if report.checkpoint_source_bound else "unbound")
        )
        lines.append(f"- **Checkpoint metadata**: {report.checkpoint_metadata_status}")
        lines.append(f"- **Checkpoint SHA-256**: `{report.checkpoint_sha256 or '(no checkpoint)'}`")
        lines.append(f"- **Model**: {report.model_name}")
        lines.append(f"- **Dimensions**: {report.dim if report.dim is not None else '?'}")
        lines.append(
            f"- **Checkpoint size**: {_human_size(report.checkpoint_size_bytes)} "
            f"({report.checkpoint_size_bytes:,} bytes)"
        )
        lines.append(
            f"- **Checkpoint mtime**: {report.checkpoint_mtime_iso or '(no checkpoint)'} (UTC)"
        )
        lines.append(f"- **Raw checkpoint lines**: {report.raw_lines:,}")
        lines.append(f"- **Distinct checkpoint chunk_ids**: {report.distinct_ids:,}")
        lines.append(f"- **Duplicate/retry lines** (raw − distinct): {report.duplicate_lines:,}")
        lines.append(f"- **Checkpoint-covered source IDs**: {report.done:,}")
        lines.append(f"- **Failed (residual, not superseded)**: {report.failed:,}")
        lines.append(f"- **Uncovered source IDs**: {report.pending:,}")
        lines.append(f"- **Total queried canonical embeddable items**: {report.total:,}")
        metric_name = (
            "Completion (source-bound checkpoint)"
            if report.checkpoint_source_bound
            else "Checkpoint ID coverage against queried generation"
        )
        lines.append(f"- **{metric_name}**: {report.coverage_pct:.2f}%")
        lines.append(f"- **Scan wall-time**: {report.elapsed_seconds:.2f}s")
        lines.append("")

    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs/generated/CORPUS_STATS.md from canonical text IDs and vectors."
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
    for registered_spec in STORES:
        spec = _env_override(registered_spec)
        print(f"[summarize_corpus] scanning store {spec.name!r}...", file=sys.stderr)
        report = _build_report(repo_root, spec)
        print(
            f"[summarize_corpus] {spec.name}: {report.done}/{report.total} IDs covered "
            f"({report.coverage_pct:.2f}%), {report.elapsed_seconds:.2f}s",
            file=sys.stderr,
        )
        reports.append(report)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render(reports), encoding="utf-8")
    store_names = ", ".join(report.spec.name for report in reports)
    print(f"Wrote {output_path} ({len(reports)} stores: {store_names})")


if __name__ == "__main__":
    main()
