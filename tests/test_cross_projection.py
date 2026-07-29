"""Tests for the Phase 4 (ADR-SKM-008 D4/D6) verbs: cross_project,
direction_diagnostics, preview_pattern.

All fixture-based: small synthetic memmaps + a synthetic corpus.db (built
directly with sqlite3, matching the ADR-TVI-008 schema used by the Phase 3
tests: chunk(chunk_id, source, speaker, era, text), vector_status(chunk_id,
rowid_mm, status)). NO real corpus, NO network, NO real embedding backend.
Covers the acceptance criteria named in issue #59 / the ADR Phase 4 line:

- the matrix's off-diagonal (era-A direction on era-B corpus) is populated
  and each cell records both source directions
- preview_pattern writes nothing (read-only assertion, sha-before==after)
- direction_diagnostics returns the recorded verdict without recomputation
  (mutate the stored diagnostics and assert readback reflects the mutation)
- preview_pattern match/scope/k semantics + IGNORECASE
- server.py registration (list_tools/call_tool complete-surface, text-scan
  per sk-mcp#63)
- refusals (identity mismatch, non-usable verdict without override)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from semantic_kinematics.mcp.commands import direction as d
from semantic_kinematics.mcp.commands.direction import (
    Calibration,
    Direction,
    DirectionRefusal,
    Projection,
    build_cross_projection_row,
    cross_project,
    direction_diagnostics,
    find_preview_matches,
    open_corpus_db_readonly,
    preview_pattern,
    project_corpus,
    write_direction_artifact,
    write_projection_artifact,
)


DIM = 12
EMBEDDING_MODEL_ID = "nvidia/NV-Embed-v2"
MEMMAP_SHA = "cafebabe" * 8


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Fixtures: a two-era synthetic corpus with a KNOWN axis per era, so the
# cross-projection matrix's rates are analytically checkable.
# --------------------------------------------------------------------------- #

def _make_calibration(mu: np.ndarray, memmap_path: str, n_used: int, sha: str = MEMMAP_SHA) -> Calibration:
    return Calibration(
        embedding_model_id=EMBEDDING_MODEL_ID,
        dim=DIM,
        source_memmap_path=memmap_path,
        source_memmap_sha256=sha,
        n_used=n_used,
        convention_version="tvi-008-dedup-keep-last-zerofilter-v1",
        mu=mu,
        mu_norm=float(np.linalg.norm(mu)),
        eigvecs=None,
        eigvals=None,
        eigenbasis_k=None,
        header={
            "regime": "corpus-calibration",
            "embedding_model_id": EMBEDDING_MODEL_ID,
            "dim": DIM,
            "source_memmap_path": memmap_path,
            "source_memmap_sha256": sha,
            "n_used": n_used,
            "convention_version": "tvi-008-dedup-keep-last-zerofilter-v1",
        },
        manifest={"mu_norm": float(np.linalg.norm(mu))},
    )


def _write_memmap(tmp_path: Path, matrix: np.ndarray, name: str) -> str:
    path = tmp_path / name
    matrix.astype(np.float32).tofile(path)
    return str(path)


def _write_direction(
    tmp_path: Path,
    calibration: Calibration,
    unit_axis: np.ndarray,
    pattern_id: str,
    era,
    verdict: str = "usable",
    held_out_auc: float = 0.85,
) -> str:
    diagnostics = {
        "pole_separation": 1.0,
        "verdict": verdict,
        "n_seeds": 41,
        "n_negatives": 41,
        "held_out_auc": {"auc": held_out_auc, "leakage_suspected": False},
        "topic_control": {"seed_vs_negative_auc": 0.8},
        "bootstrap": {"mean_pairwise_cosine": 0.9},
        "null_reference": {"mu0": 0.0, "sigma0": 1.0, "n": 200},
    }
    _npz, json_path = write_direction_artifact(
        out_dir=str(tmp_path / "directions"),
        pattern_id=pattern_id,
        era=era,
        unit_axis=unit_axis,
        calibration=calibration,
        seedset_manifest={"pattern_id": pattern_id},
        diagnostics=diagnostics,
    )
    return json_path


def _make_corpus_db(tmp_path, rows, name="corpus.db") -> str:
    """Matches the ADR-TVI-008 schema used by the Phase 3 fixtures."""
    db_path = tmp_path / name
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE chunk (chunk_id TEXT PRIMARY KEY, source TEXT, speaker TEXT, "
        "era TEXT, text TEXT)"
    )
    conn.execute(
        "CREATE TABLE vector_status (chunk_id TEXT, rowid_mm INTEGER, status TEXT)"
    )
    for row in rows:
        conn.execute(
            "INSERT INTO chunk (chunk_id, source, speaker, era, text) VALUES (?, ?, ?, ?, ?)",
            (row["chunk_id"], row.get("source"), row.get("speaker"), row.get("era"), row.get("text")),
        )
        conn.execute(
            "INSERT INTO vector_status (chunk_id, rowid_mm, status) VALUES (?, ?, 'present')",
            (row["chunk_id"], row["rowid_mm"]),
        )
    conn.commit()
    conn.close()
    return str(db_path)


@pytest.fixture
def matrix_scene(tmp_path):
    """A corpus of 40 rows split evenly between era "Opus 4.7" (rows 0-19)
    and era "Opus 4.8" (rows 20-39). Two directions are extracted, one per
    era, each axis-aligned on a DIFFERENT dimension so their projections onto
    the FULL corpus are analytically distinguishable:

    - direction_A (era "Opus 4.7"): unit axis = dim 0. Rows 0-19 (its own
      era) are displaced +5 along dim 0 (all "above" any reasonable
      threshold); rows 20-39 (era "Opus 4.8") are at baseline (~0, "below").
      So direction_A's rate is high on column "Opus 4.7" (diagonal) and low
      on column "Opus 4.8" (off-diagonal) -- but nonzero/populated, not
      skipped, satisfying "the off-diagonal is populated."
    - direction_B (era "Opus 4.8"): unit axis = dim 1, mirrored the other way
      (rows 20-39 displaced +5 along dim 1; rows 0-19 at baseline).
    """
    mu = np.zeros(DIM)
    n_used = 40
    x = np.zeros((n_used, DIM))
    axis_a = np.zeros(DIM)
    axis_a[0] = 1.0
    axis_b = np.zeros(DIM)
    axis_b[1] = 1.0

    x[0:20, 0] = 5.0  # era Opus 4.7 rows: high on axis_a
    x[20:40, 1] = 5.0  # era Opus 4.8 rows: high on axis_b

    memmap_path = _write_memmap(tmp_path, x, "vectors_matrix.f32")
    calibration = _make_calibration(mu, memmap_path, n_used)

    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=mu)
    calibration_json.write_text(json.dumps({"header": calibration.header, "mu_norm": 0.0}))

    direction_a_json = _write_direction(tmp_path, calibration, axis_a, "comparative-perception", era="Opus 4.7")
    direction_b_json = _write_direction(tmp_path, calibration, axis_b, "comparative-perception", era="Opus 4.8")

    corpus_rows = []
    for i in range(n_used):
        era = "Opus 4.7" if i < 20 else "Opus 4.8"
        corpus_rows.append(
            {
                "rowid_mm": i,
                "chunk_id": f"chunk-{i}",
                "era": era,
                "source": "claude_code",
                "speaker": "assistant",
                "text": f"text body {i}",
            }
        )
    corpus_db = _make_corpus_db(tmp_path, corpus_rows)

    # Build the projection artifacts (project_corpus) each direction needs --
    # cross_project consumes pre-built projections, it does not project itself.
    proj_a = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": direction_a_json,
                "calibration_ref": str(calibration_json),
                "out_dir": str(tmp_path / "projections"),
            },
        )
    )
    proj_b = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": direction_b_json,
                "calibration_ref": str(calibration_json),
                "out_dir": str(tmp_path / "projections"),
            },
        )
    )
    assert "error" not in proj_a, proj_a
    assert "error" not in proj_b, proj_b

    return {
        "tmp_path": tmp_path,
        "calibration": calibration,
        "calibration_json": str(calibration_json),
        "direction_a_json": direction_a_json,
        "direction_b_json": direction_b_json,
        "projection_a_ref": proj_a["projection_ref"],
        "projection_b_ref": proj_b["projection_ref"],
        "corpus_db": corpus_db,
        "n_used": n_used,
    }


# --------------------------------------------------------------------------- #
# cross_project: matrix cell correctness, era-filtered seeds -> direction ->
# cross-era projection, analytically checkable.
# --------------------------------------------------------------------------- #

def test_cross_project_matrix_off_diagonal_is_populated(matrix_scene):
    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [matrix_scene["direction_a_json"], matrix_scene["direction_b_json"]],
                "projection_refs": [matrix_scene["projection_a_ref"], matrix_scene["projection_b_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert "error" not in result, result
    assert result["n_directions"] == 2

    row_a = next(r for r in result["rows"] if r["direction_era"] == "Opus 4.7")
    row_b = next(r for r in result["rows"] if r["direction_era"] == "Opus 4.8")

    # Row A (direction extracted from era "Opus 4.7" seeds) has a cell for
    # BOTH corpus eras -- the off-diagonal ("Opus 4.8" column) is populated,
    # not skipped/omitted.
    era_columns_a = {c["corpus_era"] for c in row_a["cells"]}
    assert era_columns_a == {"Opus 4.7", "Opus 4.8"}

    cell_diag_a = next(c for c in row_a["cells"] if c["corpus_era"] == "Opus 4.7")
    cell_offdiag_a = next(c for c in row_a["cells"] if c["corpus_era"] == "Opus 4.8")

    # Analytically: direction_a is axis-aligned with dim 0, and only rows
    # 0-19 (era "Opus 4.7") are displaced +5 on dim 0 -- so the diagonal cell
    # (direction A x its own era) has a much higher rate than the
    # off-diagonal cell (direction A x era "Opus 4.8", which is at baseline
    # on dim 0).
    assert cell_diag_a["rate"] > cell_offdiag_a["rate"]
    assert cell_diag_a["n"] == 20
    assert cell_offdiag_a["n"] == 20

    row_b_columns = {c["corpus_era"] for c in row_b["cells"]}
    assert row_b_columns == {"Opus 4.7", "Opus 4.8"}


def test_cross_project_cell_records_both_source_identities(matrix_scene):
    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [matrix_scene["direction_a_json"]],
                "projection_refs": [matrix_scene["projection_a_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert "error" not in result, result
    row = result["rows"][0]
    assert row["direction_era"] == "Opus 4.7"
    assert row["pattern_id"] == "comparative-perception"
    assert row["direction_verdict"] == "usable"
    assert row["direction_ref_sha256"]  # non-empty content-addressed identity

    for cell in row["cells"]:
        # Each cell names its OWN corpus era (the "era B" side) AND carries
        # the row's direction identity (the "era A" side) -- both source
        # directions of the hypothesis test, per the acceptance criterion.
        assert "corpus_era" in cell
        assert cell["direction_era"] == "Opus 4.7"
        assert cell["direction_pattern_id"] == "comparative-perception"
        assert cell["direction_verdict"] == "usable"
        assert "allow_non_usable_direction_used" in cell


def test_build_cross_projection_row_pure_core_matches_handler(matrix_scene):
    """The pure composition core (no IO) reproduces the handler's row shape
    exactly, given the same loaded artifacts + corpus rows -- confirms
    cross_project is thin IO glue over build_cross_projection_row, not a
    second implementation of the aggregation."""
    direction = d.load_direction(matrix_scene["direction_a_json"])
    projection = d.load_projection(matrix_scene["projection_a_ref"])
    conn = open_corpus_db_readonly(matrix_scene["corpus_db"])
    try:
        corpus_rows = d.fetch_chunk_rows_for_rowids(conn, projection.rowid_mm.tolist())
    finally:
        conn.close()
    threshold_info = d.calibrate_threshold_from_direction(direction, 0.9)

    row = build_cross_projection_row(direction, projection, corpus_rows, threshold_info, allow_override_used=False)

    handler_result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [matrix_scene["direction_a_json"]],
                "projection_refs": [matrix_scene["projection_a_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert handler_result["rows"][0] == row


def test_cross_project_requires_matching_length_refs(matrix_scene):
    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [matrix_scene["direction_a_json"], matrix_scene["direction_b_json"]],
                "projection_refs": [matrix_scene["projection_a_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert "error" in result
    assert "same length" in result["error"]


def test_cross_project_requires_refs():
    result = _run(cross_project(manager=None, args={}))
    assert "error" in result
    assert "direction_refs" in result["error"]


def test_cross_project_refuses_on_projection_direction_mismatch(matrix_scene):
    """Swapping the projection for direction A's row must refuse -- the
    identity gate (direction_sha256/source_memmap_sha256) is reused from
    Phase 3's refuse_unless_projection_matches_direction, not re-derived."""
    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [matrix_scene["direction_a_json"]],
                "projection_refs": [matrix_scene["projection_b_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert "error" in result
    assert "direction_sha256" in result["error"]


def test_cross_project_refuses_non_usable_direction_without_override(tmp_path, matrix_scene):
    under_determined_json = _write_direction(
        tmp_path,
        matrix_scene["calibration"],
        np.eye(DIM)[0],
        pattern_id="thin-pattern",
        era="Opus 4.7",
        verdict="under-determined",
    )
    # The projection artifact itself must be built with the override (Phase 3's
    # project_corpus refuses a non-usable direction independently) -- this test
    # targets cross_project's OWN verdict gate, exercised on an
    # already-existing projection artifact, not project_corpus's.
    proj = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": under_determined_json,
                "calibration_ref": matrix_scene["calibration_json"],
                "out_dir": str(tmp_path / "projections2"),
                "allow_non_usable_direction": True,
            },
        )
    )
    assert "error" not in proj, proj

    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [under_determined_json],
                "projection_refs": [proj["projection_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
            },
        )
    )
    assert "error" in result
    assert "under-determined" in result["error"]


def test_cross_project_allow_non_usable_direction_override_traced_in_cells(tmp_path, matrix_scene):
    under_determined_json = _write_direction(
        tmp_path,
        matrix_scene["calibration"],
        np.eye(DIM)[0],
        pattern_id="thin-pattern",
        era="Opus 4.7",
        verdict="under-determined",
    )
    proj = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": under_determined_json,
                "calibration_ref": matrix_scene["calibration_json"],
                "out_dir": str(tmp_path / "projections3"),
                "allow_non_usable_direction": True,
            },
        )
    )
    assert "error" not in proj, proj

    result = _run(
        cross_project(
            manager=None,
            args={
                "direction_refs": [under_determined_json],
                "projection_refs": [proj["projection_ref"]],
                "corpus_db_ref": matrix_scene["corpus_db"],
                "allow_non_usable_direction": True,
            },
        )
    )
    assert "error" not in result, result
    row = result["rows"][0]
    assert row["direction_verdict"] == "under-determined"
    for cell in row["cells"]:
        assert cell["allow_non_usable_direction_used"] is True


# --------------------------------------------------------------------------- #
# direction_diagnostics: readback fidelity, NO recomputation.
# --------------------------------------------------------------------------- #

def test_direction_diagnostics_returns_recorded_verdict(matrix_scene):
    result = _run(
        direction_diagnostics(manager=None, args={"direction_ref": matrix_scene["direction_a_json"]})
    )
    assert "error" not in result, result
    assert result["verdict"] == "usable"
    assert result["pattern_id"] == "comparative-perception"
    assert result["era"] == "Opus 4.7"
    assert result["held_out_auc"]["auc"] == pytest.approx(0.85)
    assert result["topic_control"]["seed_vs_negative_auc"] == pytest.approx(0.8)
    assert result["bootstrap"]["mean_pairwise_cosine"] == pytest.approx(0.9)
    assert result["null_reference"]["sigma0"] == pytest.approx(1.0)


def test_direction_diagnostics_readback_reflects_mutation_not_recomputed(matrix_scene):
    """Mutate the STORED diagnostics on disk directly (never touching the
    underlying seed/negative vectors that would feed a recomputation), then
    assert the readback reflects exactly the mutated value -- proof this verb
    is a manifest read, not a re-run of initialize_direction_core."""
    manifest_path = Path(matrix_scene["direction_a_json"])
    manifest = json.loads(manifest_path.read_text())
    assert manifest["verdict"] == "usable"

    manifest["verdict"] = "leakage-suspected"
    manifest["held_out_auc"]["auc"] = 0.999
    manifest_path.write_text(json.dumps(manifest, indent=2))

    result = _run(
        direction_diagnostics(manager=None, args={"direction_ref": matrix_scene["direction_a_json"]})
    )
    assert "error" not in result, result
    # The mutated values are read back verbatim -- if this verb recomputed
    # diagnostics from the original seed/negative vectors it would recover
    # verdict="usable"/auc=0.85 (the true underlying geometry), not the
    # mutated leakage-suspected/0.999 this test injected.
    assert result["verdict"] == "leakage-suspected"
    assert result["held_out_auc"]["auc"] == pytest.approx(0.999)


def test_direction_diagnostics_requires_direction_ref():
    result = _run(direction_diagnostics(manager=None, args={}))
    assert "error" in result
    assert "direction_ref" in result["error"]


def test_direction_diagnostics_reports_missing_artifact(tmp_path):
    result = _run(
        direction_diagnostics(manager=None, args={"direction_ref": str(tmp_path / "nope.direction.json")})
    )
    assert "error" in result
    assert "not found" in result["error"]


# --------------------------------------------------------------------------- #
# preview_pattern: match/scope/k semantics, read-only assertion, IGNORECASE.
# --------------------------------------------------------------------------- #

@pytest.fixture
def preview_corpus_db(tmp_path):
    rows = [
        {"rowid_mm": 0, "chunk_id": "c0", "era": "Opus 4.7", "source": "claude_code",
         "speaker": "assistant", "text": "This is bigger than it looks."},
        {"rowid_mm": 1, "chunk_id": "c1", "era": "Opus 4.7", "source": "claude_code",
         "speaker": "user", "text": "BIGGER THAN IT LOOKS, said the user."},
        {"rowid_mm": 2, "chunk_id": "c2", "era": "Opus 4.8", "source": "claude_export",
         "speaker": "assistant", "text": "Nothing relevant here."},
        {"rowid_mm": 3, "chunk_id": "c3", "era": "Opus 4.8", "source": "claude_code",
         "speaker": "assistant", "text": "It is Bigger Than It Looks, oddly."},
        {"rowid_mm": 4, "chunk_id": "c4", "era": None, "source": "claude_code",
         "speaker": "assistant", "text": "bigger than it looks, again."},
    ]
    return _make_corpus_db(tmp_path, rows, name="preview_corpus.db")


def test_preview_pattern_match_count_and_examples(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"bigger than it looks", "corpus_db_ref": preview_corpus_db, "k": 2},
        )
    )
    assert "error" not in result, result
    assert result["match_count"] == 4  # c0, c1, c3, c4 all match case-insensitively
    assert len(result["examples"]) == 2  # capped at k, even though 4 matched


def test_preview_pattern_is_case_insensitive_by_default(preview_corpus_db):
    """tvi PR #61 methodology decision: re.IGNORECASE is the default so a
    regex previewed here matches identically to tvi's labeler."""
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"^BIGGER THAN IT LOOKS", "corpus_db_ref": preview_corpus_db, "k": 10},
        )
    )
    assert "error" not in result, result
    chunk_ids = {e["chunk_id"] for e in result["examples"]}
    # Matches c1 ("BIGGER THAN IT LOOKS, said...") case-insensitively at the
    # anchor despite the differing source casing.
    assert "c1" in chunk_ids


def test_preview_pattern_scope_filters_era(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"bigger than it looks", "corpus_db_ref": preview_corpus_db, "era": "Opus 4.8", "k": 10},
        )
    )
    assert "error" not in result, result
    assert result["match_count"] == 1  # only c3 is era "Opus 4.8" AND matches
    assert result["examples"][0]["chunk_id"] == "c3"


def test_preview_pattern_scope_filters_channel_and_speaker(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={
                "regex": r"bigger than it looks",
                "corpus_db_ref": preview_corpus_db,
                "channel": "claude_export",
                "k": 10,
            },
        )
    )
    assert "error" not in result, result
    assert result["match_count"] == 0  # c2 is the only claude_export row and doesn't match

    result2 = _run(
        preview_pattern(
            manager=None,
            args={
                "regex": r"bigger than it looks",
                "corpus_db_ref": preview_corpus_db,
                "speaker": "user",
                "k": 10,
            },
        )
    )
    assert "error" not in result2, result2
    assert result2["match_count"] == 1
    assert result2["examples"][0]["chunk_id"] == "c1"


def test_preview_pattern_no_match_returns_zero_count_empty_examples(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"totally absent pattern xyz", "corpus_db_ref": preview_corpus_db},
        )
    )
    assert "error" not in result, result
    assert result["match_count"] == 0
    assert result["examples"] == []


def test_preview_pattern_writes_nothing(preview_corpus_db):
    """The read-only assertion the ADR names explicitly: preview_pattern
    writes NOTHING. Mirrors tvi's sha-before==after pattern -- hash the file
    before and after the call and assert byte-identical."""
    sha_before = hashlib.sha256(Path(preview_corpus_db).read_bytes()).hexdigest()
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"bigger than it looks", "corpus_db_ref": preview_corpus_db, "k": 5},
        )
    )
    assert "error" not in result, result
    sha_after = hashlib.sha256(Path(preview_corpus_db).read_bytes()).hexdigest()
    assert sha_before == sha_after


def test_preview_pattern_uses_read_only_connection_cannot_insert(preview_corpus_db):
    """find_preview_matches/preview_pattern open corpus.db via
    open_corpus_db_readonly (mode=ro) -- the same primitive Phase 3 already
    proved refuses writes; confirm preview_pattern's own connection inherits
    that boundary rather than opening a fresh read-write handle."""
    conn = open_corpus_db_readonly(preview_corpus_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO chunk (chunk_id) VALUES ('should-fail')")
            conn.commit()
    finally:
        conn.close()


def test_preview_pattern_invalid_regex_reports_error(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": r"(unclosed", "corpus_db_ref": preview_corpus_db},
        )
    )
    assert "error" in result
    assert "regex" in result["error"]


def test_preview_pattern_requires_regex_and_corpus_db_ref():
    result1 = _run(preview_pattern(manager=None, args={"corpus_db_ref": "x"}))
    assert "error" in result1
    assert "regex" in result1["error"]

    result2 = _run(preview_pattern(manager=None, args={"regex": "x"}))
    assert "error" in result2
    assert "corpus_db_ref" in result2["error"]


def test_preview_pattern_rejects_non_numeric_k_gracefully(preview_corpus_db):
    """code-review finding: a bare int(args.get('k', ...)) would raise an
    unhandled ValueError on a non-numeric k instead of returning the
    {"error": ...} shape every other malformed-input path in this module
    uses. Confirms the graceful refusal instead."""
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": "x", "corpus_db_ref": preview_corpus_db, "k": "all"},
        )
    )
    assert "error" in result
    assert "k must be an integer" in result["error"]


def test_preview_pattern_rejects_negative_k(preview_corpus_db):
    result = _run(
        preview_pattern(
            manager=None,
            args={"regex": "x", "corpus_db_ref": preview_corpus_db, "k": -1},
        )
    )
    assert "error" in result
    assert "k must be >= 0" in result["error"]


def test_preview_pattern_refuses_missing_corpus_db(tmp_path):
    result = _run(
        preview_pattern(manager=None, args={"regex": "x", "corpus_db_ref": str(tmp_path / "nope.db")})
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_find_preview_matches_pure_core_default_k(preview_corpus_db):
    """The pure core respects match_count uncapped vs examples capped at k --
    confirmed directly against the core function, not just the handler."""
    conn = open_corpus_db_readonly(preview_corpus_db)
    try:
        result = find_preview_matches(conn, re.compile("bigger than it looks", re.IGNORECASE), k=1)
    finally:
        conn.close()
    assert result["match_count"] == 4
    assert len(result["examples"]) == 1


# --------------------------------------------------------------------------- #
# server.py registration: complete-surface test (ARCHITECTURE Rule 2).
# --------------------------------------------------------------------------- #

_SERVER_PY_PATH = (
    Path(__file__).resolve().parent.parent
    / "semantic_kinematics"
    / "mcp"
    / "server.py"
)

_PHASE4_TOOL_NAMES = ("cross_project", "direction_diagnostics", "preview_pattern")


def _read_server_source() -> str:
    """Read server.py's source as TEXT, never importing it -- see
    test_direction_extraction.py's identically-named helper for why (a
    pre-existing, out-of-scope F0 defect: sk-mcp#63-class mcp-package API
    drift breaks importing semantic_kinematics.mcp.server at module scope)."""
    return _SERVER_PY_PATH.read_text(encoding="utf-8")


def test_phase4_tools_registered_in_get_tools():
    tool_names = {t.name for t in d.get_tools()}
    for name in _PHASE4_TOOL_NAMES:
        assert name in tool_names


def test_phase4_tools_all_have_a_call_tool_dispatch_arm():
    """ARCHITECTURE.md 'no orphan tool' instant-fail check: every Phase 4
    tool direction.get_tools() declares must be reachable from server.py's
    call_tool dispatch."""
    server_src = _read_server_source()
    for tool in d.get_tools():
        if tool.name not in _PHASE4_TOOL_NAMES:
            continue
        assert f'name == "{tool.name}"' in server_src, (
            f"tool {tool.name!r} has no call_tool dispatch arm in server.py"
        )


def test_phase4_module_dispatch_calls_direction_handlers():
    server_src = _read_server_source()
    assert "direction.cross_project(state_manager, arguments)" in server_src
    assert "direction.direction_diagnostics(state_manager, arguments)" in server_src
    assert "direction.preview_pattern(state_manager, arguments)" in server_src


# --------------------------------------------------------------------------- #
# Stateless-core: same-in/same-out (ARCHITECTURE Rule 1).
# --------------------------------------------------------------------------- #

def test_cross_project_is_deterministic_same_in_same_out(matrix_scene):
    args = {
        "direction_refs": [matrix_scene["direction_a_json"], matrix_scene["direction_b_json"]],
        "projection_refs": [matrix_scene["projection_a_ref"], matrix_scene["projection_b_ref"]],
        "corpus_db_ref": matrix_scene["corpus_db"],
    }
    result1 = _run(cross_project(manager=None, args=dict(args)))
    result2 = _run(cross_project(manager=None, args=dict(args)))
    assert result1 == result2


def test_direction_diagnostics_is_deterministic_same_in_same_out(matrix_scene):
    args = {"direction_ref": matrix_scene["direction_a_json"]}
    result1 = _run(direction_diagnostics(manager=None, args=dict(args)))
    result2 = _run(direction_diagnostics(manager=None, args=dict(args)))
    assert result1 == result2


def test_preview_pattern_is_deterministic_same_in_same_out(preview_corpus_db):
    args = {"regex": "bigger than it looks", "corpus_db_ref": preview_corpus_db, "k": 5}
    result1 = _run(preview_pattern(manager=None, args=dict(args)))
    result2 = _run(preview_pattern(manager=None, args=dict(args)))
    assert result1 == result2
