"""Tests for the projection + rate primitives (ADR-SKM-008 Phase 3, D4/D5):
project_text, project_chunks, project_corpus, query_rates, top_exemplars.

All fixture-based: small synthetic memmaps + a synthetic corpus.db (built with
sqlite3 directly, matching the ADR-TVI-008 schema: chunk(chunk_id, source,
speaker, era, text, ...), vector_status(chunk_id, rowid_mm, ...)). NO real
corpus, NO network, NO real embedding backend (project_text uses a faked
adapter via a stub StateManager). Covers the acceptance criteria named in
issue #58:

- project_corpus matvec reconciles row count to the memmap n_used
- a rate table's manifest records threshold + precision + source shas
  (regen-not-authored)
- top_exemplars fetches text by chunk_id via corpus.db read-only
- project_text refuses a non-usable direction (generalized to every
  project_* verb)
- NULL-era rows are a first-class rate bucket, never dropped
- server.py registration (complete-surface, text-scan per sk-mcp#63)
- statelessness (same-in/same-out)
"""

from __future__ import annotations

import asyncio
import json
import os
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
    build_rate_table,
    calibrate_threshold_from_direction,
    cosine_to_axis,
    fetch_chunk_id_for_rowid,
    fetch_chunk_rows_for_rowids,
    fetch_chunk_text,
    load_projection,
    open_corpus_db_readonly,
    project_chunks,
    project_corpus,
    project_text,
    project_vectors,
    query_rates,
    refuse_unless_projection_matches_direction,
    refuse_unless_usable,
    resolve_memmap_path,
    top_exemplars,
    write_direction_artifact,
    write_projection_artifact,
    write_rate_table_artifact,
    z_scores,
)


DIM = 12
EMBEDDING_MODEL_ID = "nvidia/NV-Embed-v2"
MEMMAP_SHA = "cafebabe" * 8


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Fixtures: synthetic memmap + calibration + direction (known geometry) +
# corpus.db (ADR-TVI-008 schema).
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


def _write_memmap(tmp_path: Path, matrix: np.ndarray, name: str = "vectors_test.f32") -> str:
    path = tmp_path / name
    matrix.astype(np.float32).tofile(path)
    return str(path)


def _write_direction_artifact(
    tmp_path: Path,
    calibration: Calibration,
    unit_axis: np.ndarray,
    pattern_id: str = "comparative-perception",
    era=None,
    verdict: str = "usable",
    null_mu0: float = 0.0,
    null_sigma0: float = 1.0,
    held_out_auc: float = 0.85,
    held_out_seed_proj: np.ndarray = None,
    held_out_negative_proj: np.ndarray = None,
) -> str:
    held_out_auc_block = {"auc": held_out_auc, "leakage_suspected": False}
    if held_out_seed_proj is not None:
        held_out_auc_block["held_out_seed_proj"] = held_out_seed_proj
    if held_out_negative_proj is not None:
        held_out_auc_block["held_out_negative_proj"] = held_out_negative_proj
    diagnostics = {
        "pole_separation": 1.0,
        "verdict": verdict,
        "n_seeds": 41,
        "n_negatives": 41,
        "held_out_auc": held_out_auc_block,
        "topic_control": {"seed_vs_negative_auc": 0.8},
        "bootstrap": {"mean_pairwise_cosine": 0.9},
        "null_reference": {"mu0": null_mu0, "sigma0": null_sigma0, "n": 200},
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


@pytest.fixture
def scene(tmp_path):
    """A synthetic corpus with a KNOWN axis: row i's projection onto
    true_axis is analytically ``i * step`` (before centering), so
    project_corpus/project_chunks results are exactly checkable."""
    rng = np.random.default_rng(13)
    mu = np.zeros(DIM)
    true_axis = np.zeros(DIM)
    true_axis[0] = 1.0  # axis-aligned with dim 0 for exact arithmetic

    n_used = 20
    x = rng.normal(size=(n_used, DIM)) * 0.01  # tiny orthogonal noise
    # Row i displaced along true_axis by i (0, 1, 2, ..., 19).
    displacements = np.arange(n_used, dtype=np.float64)
    x[:, 0] = displacements

    memmap_path = _write_memmap(tmp_path, x)
    calibration = _make_calibration(mu, memmap_path, n_used)
    direction_json = _write_direction_artifact(tmp_path, calibration, true_axis)

    calibration_json = tmp_path / "cal.calibration.json"
    calibration_npz = tmp_path / "cal.calibration.npz"
    np.savez(calibration_npz, mu=mu)
    calibration_json.write_text(json.dumps({"header": calibration.header, "mu_norm": 0.0}))

    return {
        "rng": rng,
        "mu": mu,
        "true_axis": true_axis,
        "x": x,
        "displacements": displacements,
        "memmap_path": memmap_path,
        "n_used": n_used,
        "calibration": calibration,
        "calibration_json": str(calibration_json),
        "direction_json": direction_json,
    }


def _make_corpus_db(tmp_path, rows) -> str:
    """Build a synthetic corpus.db matching the ADR-TVI-008 schema:
    chunk(chunk_id, source, speaker, era, text, ...) + vector_status(chunk_id,
    rowid_mm, ...). ``rows`` is a list of dicts with keys rowid_mm, chunk_id,
    era (may be None), source, speaker, text.
    """
    db_path = tmp_path / "corpus.db"
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
def corpus_db(tmp_path, scene):
    """A corpus.db covering all n_used rows in `scene`, with a mix of eras
    (including NULL) and channels/speakers, so rate-table grouping is
    checkable including the NULL-era bucket."""
    rows = []
    eras = ["Opus 4.8", "Opus 4.7", "Gemini", None]
    sources = ["claude_code", "claude_export"]
    speakers = ["assistant", "user"]
    for i in range(scene["n_used"]):
        rows.append(
            {
                "rowid_mm": i,
                "chunk_id": f"chunk-{i}",
                "era": eras[i % len(eras)],
                "source": sources[i % len(sources)],
                "speaker": speakers[i % len(speakers)],
                "text": f"text body {i}",
            }
        )
    return _make_corpus_db(tmp_path, rows)


class _FakeAdapter:
    """A faked embedding adapter -- no live server, per the dispatch's
    'text path with a faked adapter' instruction. Returns a vector whose
    dim-0 component is controlled by the caller (via a lookup keyed on text),
    so project_text's projection is exactly checkable."""

    def __init__(self, model_name: str, text_to_dim0: dict):
        self.model_name = model_name
        self.dimensions = DIM
        self._text_to_dim0 = text_to_dim0

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(DIM)
        vec[0] = self._text_to_dim0.get(text, 0.0)
        return vec


class _FakeManager:
    """Stub StateManager exposing only get_adapter(), matching what
    project_text's handler calls (ARCHITECTURE Rule 1 stateless-core: no
    cross-call state needed beyond the adapter handle)."""

    def __init__(self, adapter):
        self._adapter = adapter

    def get_adapter(self):
        return self._adapter


# --------------------------------------------------------------------------- #
# Pure numeric core: project_vectors, z_scores, cosine_to_axis.
# --------------------------------------------------------------------------- #

def test_project_vectors_is_plain_matvec():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 3.0]])
    axis = np.array([1.0, 0.0])
    np.testing.assert_allclose(project_vectors(vecs, axis), [1.0, 0.0, 2.0])


def test_z_scores_normalizes_against_null():
    proj = np.array([0.0, 1.0, 2.0, 3.0])
    z = z_scores(proj, mu0=1.0, sigma0=2.0)
    np.testing.assert_allclose(z, [-0.5, 0.0, 0.5, 1.0])


def test_z_scores_refuses_zero_sigma():
    with pytest.raises(DirectionRefusal, match="sigma0"):
        z_scores(np.array([1.0]), mu0=0.0, sigma0=0.0)


def test_cosine_to_axis_unit_vector_along_axis_is_one():
    axis = np.array([1.0, 0.0, 0.0])
    vec = np.array([5.0, 0.0, 0.0])
    assert cosine_to_axis(vec, axis) == pytest.approx(1.0)


def test_cosine_to_axis_orthogonal_is_zero():
    axis = np.array([1.0, 0.0])
    vec = np.array([0.0, 3.0])
    assert cosine_to_axis(vec, axis) == pytest.approx(0.0)


def test_cosine_to_axis_zero_vector_is_zero_not_nan():
    axis = np.array([1.0, 0.0])
    assert cosine_to_axis(np.zeros(2), axis) == 0.0


# --------------------------------------------------------------------------- #
# resolve_memmap_path: shared tvi_root resolution (extracted, not duplicated).
# --------------------------------------------------------------------------- #

def test_resolve_memmap_path_absolute_passthrough():
    assert resolve_memmap_path("/abs/path.f32", None) == "/abs/path.f32"


def test_resolve_memmap_path_relative_with_tvi_root():
    assert resolve_memmap_path("output/x.f32", "/tvi") == "/tvi/output/x.f32"


def test_resolve_memmap_path_relative_without_tvi_root_refuses(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DirectionRefusal, match="tvi_root"):
        resolve_memmap_path("does/not/exist.f32", None)


# --------------------------------------------------------------------------- #
# refuse_unless_usable: the D3/D5 verdict gate, generalized to every project_*.
# --------------------------------------------------------------------------- #

def _direction_with_verdict(verdict: str) -> Direction:
    return Direction(
        embedding_model_id=EMBEDDING_MODEL_ID,
        source_memmap_sha256=MEMMAP_SHA,
        pattern_id="p",
        era=None,
        dim=DIM,
        unit_axis=np.ones(DIM) / np.sqrt(DIM),
        axis_source="seedset_centroids",
        pole_separation=1.0,
        verdict=verdict,
        direction_sha256="deadbeef",
    )


def test_refuse_unless_usable_passes_on_usable():
    refuse_unless_usable(_direction_with_verdict("usable"), allow_override=False)  # no raise


@pytest.mark.parametrize("verdict", ["under-determined", "leakage-suspected"])
def test_refuse_unless_usable_refuses_non_usable_by_default(verdict):
    with pytest.raises(DirectionRefusal, match=verdict):
        refuse_unless_usable(_direction_with_verdict(verdict), allow_override=False)


@pytest.mark.parametrize("verdict", ["under-determined", "leakage-suspected"])
def test_refuse_unless_usable_override_allows_non_usable(verdict):
    refuse_unless_usable(_direction_with_verdict(verdict), allow_override=True)  # no raise


# --------------------------------------------------------------------------- #
# project_corpus: full-matvec, artifact persistence, row-count reconciliation
# (the ADR's named acceptance criterion), refusal chain.
# --------------------------------------------------------------------------- #

def test_project_corpus_reconciles_row_count_to_memmap_n_used(tmp_path, scene):
    out_dir = tmp_path / "projections"
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(out_dir),
            },
        )
    )
    assert "error" not in result, result
    assert result["n_rows"] == scene["n_used"]
    assert result["n_used"] == scene["n_used"]

    loaded = load_projection(result["projection_ref"])
    assert loaded.rowid_mm.shape[0] == scene["n_used"]
    assert loaded.n_used == scene["n_used"]


def test_project_corpus_matvec_matches_known_axis_displacement(tmp_path, scene):
    """The scene's true_axis is dim-0-aligned and mu=0, so the projection of
    row i is EXACTLY its displacement (i), analytically checkable."""
    out_dir = tmp_path / "projections"
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(out_dir),
            },
        )
    )
    loaded = load_projection(result["projection_ref"])
    # The persisted z is the z-score of the raw projection against the
    # direction's null_reference; the default fixture's null is mu0=0,sigma0=1,
    # so z == raw projection here (the raw value is also persisted separately as
    # loaded.projection -- both checked against the analytic displacement).
    by_rowid = dict(zip(loaded.rowid_mm.tolist(), loaded.z.tolist()))
    by_rowid_raw = dict(zip(loaded.rowid_mm.tolist(), loaded.projection.tolist()))
    for i, expected in enumerate(scene["displacements"]):
        assert by_rowid[i] == pytest.approx(expected, abs=1e-4)
        assert by_rowid_raw[i] == pytest.approx(expected, abs=1e-4)


def test_project_corpus_persists_direction_and_memmap_sha(tmp_path, scene):
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections"),
            },
        )
    )
    manifest = json.loads(Path(result["projection_ref"]).read_text())
    assert manifest["header"]["source_memmap_sha256"] == scene["calibration"].source_memmap_sha256
    assert manifest["header"]["direction_sha256"]  # non-empty


def test_project_corpus_refuses_under_determined_direction_by_default(tmp_path, scene):
    under_determined_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="thin-pattern",
        verdict="under-determined",
    )
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": under_determined_direction,
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections"),
            },
        )
    )
    assert "error" in result
    assert "under-determined" in result["error"]


def test_project_corpus_override_allows_under_determined_direction(tmp_path, scene):
    under_determined_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="thin-pattern",
        verdict="under-determined",
    )
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": under_determined_direction,
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections"),
                d.ALLOW_NON_USABLE_PROJECTION_ARG: True,
            },
        )
    )
    assert "error" not in result, result
    assert result["allow_non_usable_direction_used"] is True


def test_project_corpus_requires_refs():
    result = _run(project_corpus(manager=None, args={}))
    assert "error" in result
    assert "direction_ref" in result["error"]


# --------------------------------------------------------------------------- #
# project_chunks: caller-supplied rowid_mm subset, no embedding call.
# --------------------------------------------------------------------------- #

def test_project_chunks_projects_only_requested_rows(tmp_path, scene):
    result = _run(
        project_chunks(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "rowids": [0, 5, 10],
            },
        )
    )
    assert "error" not in result, result
    rows = {r["rowid_mm"]: r["z"] for r in result["rows"]}
    assert set(rows.keys()) == {0, 5, 10}
    assert rows[5] == pytest.approx(5.0, abs=1e-4)
    assert rows[10] == pytest.approx(10.0, abs=1e-4)


def test_project_chunks_rejects_out_of_range_rowid(tmp_path, scene):
    result = _run(
        project_chunks(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "rowids": [scene["n_used"] + 100],
            },
        )
    )
    assert "error" in result
    assert "out of range" in result["error"]


def test_project_chunks_refuses_non_usable_direction(tmp_path, scene):
    leakage_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="leaky",
        verdict="leakage-suspected",
    )
    result = _run(
        project_chunks(
            manager=None,
            args={
                "direction_ref": leakage_direction,
                "calibration_ref": scene["calibration_json"],
                "rowids": [0, 1],
            },
        )
    )
    assert "error" in result
    assert "leakage-suspected" in result["error"]


def test_project_chunks_override_allows_non_usable_direction(tmp_path, scene):
    leakage_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="leaky2",
        verdict="leakage-suspected",
    )
    result = _run(
        project_chunks(
            manager=None,
            args={
                "direction_ref": leakage_direction,
                "calibration_ref": scene["calibration_json"],
                "rowids": [0, 1],
                d.ALLOW_NON_USABLE_PROJECTION_ARG: True,
            },
        )
    )
    assert "error" not in result, result
    assert result["allow_non_usable_direction_used"] is True


def test_project_chunks_requires_rowids():
    result = _run(
        project_chunks(manager=None, args={"direction_ref": "x", "calibration_ref": "y"})
    )
    assert "error" in result
    assert "rowids" in result["error"]


# --------------------------------------------------------------------------- #
# project_text: faked adapter, no live embedding server.
# --------------------------------------------------------------------------- #

def test_project_text_projects_embedded_vector(scene):
    adapter = _FakeAdapter(EMBEDDING_MODEL_ID, {"the passage": 7.0})
    manager = _FakeManager(adapter)
    result = _run(
        project_text(
            manager=manager,
            args={
                "text": "the passage",
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
            },
        )
    )
    assert "error" not in result, result
    assert result["projection"] == pytest.approx(7.0, abs=1e-4)
    assert result["cosine"] == pytest.approx(1.0, abs=1e-3)
    # null_reference in the fixture direction is mu0=0, sigma0=1 -> z == projection.
    assert result["z"] == pytest.approx(7.0, abs=1e-4)


def test_project_text_refuses_non_usable_direction(tmp_path, scene):
    thin_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="thin",
        verdict="under-determined",
    )
    adapter = _FakeAdapter(EMBEDDING_MODEL_ID, {"x": 1.0})
    manager = _FakeManager(adapter)
    result = _run(
        project_text(
            manager=manager,
            args={"text": "x", "direction_ref": thin_direction, "calibration_ref": scene["calibration_json"]},
        )
    )
    assert "error" in result
    assert "under-determined" in result["error"]


def test_project_text_override_allows_non_usable_direction(tmp_path, scene):
    thin_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="thin",
        verdict="under-determined",
    )
    adapter = _FakeAdapter(EMBEDDING_MODEL_ID, {"x": 1.0})
    manager = _FakeManager(adapter)
    result = _run(
        project_text(
            manager=manager,
            args={
                "text": "x",
                "direction_ref": thin_direction,
                "calibration_ref": scene["calibration_json"],
                d.ALLOW_NON_USABLE_PROJECTION_ARG: True,
            },
        )
    )
    assert "error" not in result, result
    assert result["allow_non_usable_direction_used"] is True


def test_project_text_refuses_model_mismatch(scene):
    adapter = _FakeAdapter("some-other-model", {"x": 1.0})
    manager = _FakeManager(adapter)
    result = _run(
        project_text(
            manager=manager,
            args={"text": "x", "direction_ref": scene["direction_json"], "calibration_ref": scene["calibration_json"]},
        )
    )
    assert "error" in result
    assert "embedding_model_id" in result["error"] or "embedding space" in result["error"]


def test_project_text_requires_text():
    result = _run(
        project_text(manager=None, args={"direction_ref": "x", "calibration_ref": "y"})
    )
    assert "error" in result
    assert "text" in result["error"]


# --------------------------------------------------------------------------- #
# corpus.db read-only helpers.
# --------------------------------------------------------------------------- #

def test_open_corpus_db_readonly_opens_existing_file(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        cur = conn.execute("SELECT COUNT(*) FROM chunk")
        assert cur.fetchone()[0] > 0
    finally:
        conn.close()


def test_open_corpus_db_readonly_refuses_missing_file(tmp_path):
    with pytest.raises(DirectionRefusal, match="corpus.db not found"):
        open_corpus_db_readonly(str(tmp_path / "nope.db"))


def test_open_corpus_db_readonly_cannot_write(corpus_db):
    """The read-only boundary (ADR-TVI-008/ADR-SKM-008): opening via mode=ro
    means an INSERT attempt raises, never silently succeeds."""
    conn = open_corpus_db_readonly(corpus_db)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO chunk (chunk_id) VALUES ('should-fail')")
            conn.commit()
    finally:
        conn.close()


def test_fetch_chunk_rows_for_rowids_joins_era_channel_speaker(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        rows = fetch_chunk_rows_for_rowids(conn, [0, 1, 2, 3])
    finally:
        conn.close()
    assert len(rows) == 4
    by_rowid = {r["rowid_mm"]: r for r in rows}
    assert by_rowid[3]["era"] is None  # 4th row (i=3) is the NULL-era slot per fixture
    assert by_rowid[0]["channel"] == "claude_code"
    assert by_rowid[0]["speaker"] == "assistant"


def test_fetch_chunk_rows_for_rowids_empty_list_returns_empty(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        assert fetch_chunk_rows_for_rowids(conn, []) == []
    finally:
        conn.close()


def test_fetch_chunk_text_by_chunk_id(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        texts = fetch_chunk_text(conn, ["chunk-0", "chunk-1"])
    finally:
        conn.close()
    assert texts["chunk-0"] == "text body 0"
    assert texts["chunk-1"] == "text body 1"


def test_fetch_chunk_id_for_rowid(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        mapping = fetch_chunk_id_for_rowid(conn, [0, 1])
    finally:
        conn.close()
    assert mapping == {0: "chunk-0", 1: "chunk-1"}


# --------------------------------------------------------------------------- #
# build_rate_table: pure aggregation, NULL-era bucket kept, not dropped.
# --------------------------------------------------------------------------- #

def _make_projection(rowid_mm, z, pattern_id="p", era=None) -> Projection:
    return Projection(
        embedding_model_id=EMBEDDING_MODEL_ID,
        source_memmap_sha256=MEMMAP_SHA,
        direction_sha256="deadbeef",
        pattern_id=pattern_id,
        era=era,
        n_used=len(rowid_mm),
        rowid_mm=np.asarray(rowid_mm, dtype=np.int64),
        z=np.asarray(z, dtype=np.float64),
    )


def test_build_rate_table_computes_fraction_above_threshold():
    projection = _make_projection([0, 1, 2, 3], [0.1, 5.0, 6.0, 0.2])
    corpus_rows = [
        {"rowid_mm": 0, "era": "Opus 4.8", "channel": "claude_code", "speaker": "assistant"},
        {"rowid_mm": 1, "era": "Opus 4.8", "channel": "claude_code", "speaker": "assistant"},
        {"rowid_mm": 2, "era": "Opus 4.8", "channel": "claude_code", "speaker": "assistant"},
        {"rowid_mm": 3, "era": "Opus 4.8", "channel": "claude_code", "speaker": "assistant"},
    ]
    table = build_rate_table(projection, threshold=3.0, corpus_rows=corpus_rows)
    assert table["n_total"] == 4
    assert len(table["groups"]) == 1
    group = table["groups"][0]
    assert group["n"] == 4
    assert group["n_above"] == 2
    assert group["rate"] == pytest.approx(0.5)


def test_build_rate_table_keeps_null_era_as_first_class_bucket():
    projection = _make_projection([0, 1], [10.0, 10.0])
    corpus_rows = [
        {"rowid_mm": 0, "era": None, "channel": "claude_code", "speaker": "assistant"},
        {"rowid_mm": 1, "era": "Gemini", "channel": "gemini_exporter", "speaker": "assistant"},
    ]
    table = build_rate_table(projection, threshold=1.0, corpus_rows=corpus_rows)
    keys = [g["key"]["era"] for g in table["groups"]]
    assert "NULL" in keys  # NULL era not silently dropped
    assert "Gemini" in keys
    null_group = next(g for g in table["groups"] if g["key"]["era"] == "NULL")
    assert null_group["n"] == 1
    assert null_group["n_above"] == 1


def test_build_rate_table_groups_by_subset_of_fields():
    projection = _make_projection([0, 1], [10.0, -10.0])
    corpus_rows = [
        {"rowid_mm": 0, "era": "Opus 4.8", "channel": "a", "speaker": "assistant"},
        {"rowid_mm": 1, "era": "Opus 4.8", "channel": "b", "speaker": "user"},
    ]
    table = build_rate_table(projection, threshold=0.0, corpus_rows=corpus_rows, group_by=["era"])
    assert len(table["groups"]) == 1  # both rows share era -> one group
    assert table["groups"][0]["n"] == 2
    assert table["groups"][0]["n_above"] == 1


def test_build_rate_table_ignores_rows_not_in_projection():
    """A corpus.db join can return rows the projection artifact does not
    cover (e.g. a subsequent memmap append); such rows must not corrupt the
    rate table with an unmapped rowid_mm."""
    projection = _make_projection([0], [10.0])
    corpus_rows = [
        {"rowid_mm": 0, "era": "Opus 4.8", "channel": "a", "speaker": "assistant"},
        {"rowid_mm": 999, "era": "Opus 4.8", "channel": "a", "speaker": "assistant"},
    ]
    table = build_rate_table(projection, threshold=0.0, corpus_rows=corpus_rows)
    assert table["n_total"] == 1


# --------------------------------------------------------------------------- #
# calibrate_threshold_from_direction (D5).
# --------------------------------------------------------------------------- #

def test_calibrate_threshold_from_direction_uses_held_out_auc(tmp_path, scene):
    direction = d.load_direction(scene["direction_json"])
    threshold_info = calibrate_threshold_from_direction(direction, target_precision=0.9)
    assert threshold_info["source_auc"] == pytest.approx(0.85)
    assert threshold_info["target_precision"] == 0.9
    assert isinstance(threshold_info["threshold"], float)


def test_calibrate_threshold_refuses_without_held_out_auc(tmp_path, scene):
    direction = d.load_direction(scene["direction_json"])
    direction.manifest["held_out_auc"] = {}
    with pytest.raises(DirectionRefusal, match="held_out_auc"):
        calibrate_threshold_from_direction(direction)


def test_calibrate_threshold_higher_auc_yields_higher_threshold(tmp_path, scene):
    direction = d.load_direction(scene["direction_json"])
    direction.manifest["held_out_auc"] = {"auc": 0.99}
    high = calibrate_threshold_from_direction(direction, target_precision=0.9)
    direction.manifest["held_out_auc"] = {"auc": 0.55}
    low = calibrate_threshold_from_direction(direction, target_precision=0.9)
    assert high["threshold"] > low["threshold"]


def test_calibrate_threshold_uses_exact_curve_when_raw_projections_present(tmp_path):
    """issue #66: when the direction artifact carries the raw held-out
    projection arrays, calibrate_threshold_from_direction computes the exact
    D5 empirical precision-recall threshold instead of the gaussian-quantile
    approximation."""
    mu = np.zeros(DIM)
    calibration = _make_calibration(mu, "unused.f32", 20)
    unit_axis = np.zeros(DIM)
    unit_axis[0] = 1.0

    # Hand-built held-out projections (already in raw-projection units;
    # null_reference mu0=0/sigma0=1 below makes z == raw projection).
    # Seeds: 1, 2, 3, 4, 5.  Negatives: -2, -1, 0, 0.5, 3.5.
    #
    # _exact_precision_threshold now honors its declared contract: it returns
    # the *highest* z at which precision first reaches target, scanning the
    # distinct candidate z's high->low and returning on the first qualifier.
    # The full precision curve over this fixture (candidates high->low) is
    # non-monotone:
    #     z=5.0  seeds>= {5}=1        negs>= {}=0      precision=1/1 = 1.000  (QUALIFIES)
    #     z=4.0  seeds>= {4,5}=2      negs>= {}=0      precision=2/2 = 1.000
    #     z=3.5  seeds>= {4,5}=2      negs>= {3.5}=1   precision=2/3 = 0.667
    #     z=3.0  seeds>= {3,4,5}=3    negs>= {3.5}=1   precision=3/4 = 0.750
    #     z=2.0  seeds>= {2,3,4,5}=4  negs>= {3.5}=1   precision=4/5 = 0.800  (re-qualifies)
    #     z=1.0  seeds>= all=5        negs>= {3.5}=1   precision=5/6 = 0.833
    # The highest qualifying z is therefore z=5.0 -- the scan returns on that
    # first hit and never reaches the z=2 re-qualification. (Pre-fix, the code
    # kept scanning to the *lowest* qualifier and returned z=1.0; that bug is
    # what this assertion pins.)
    seed_proj = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    neg_proj = np.array([-2.0, -1.0, 0.0, 0.5, 3.5])

    direction_json = _write_direction_artifact(
        tmp_path, calibration, unit_axis,
        null_mu0=0.0, null_sigma0=1.0, held_out_auc=0.8,
        held_out_seed_proj=seed_proj, held_out_negative_proj=neg_proj,
    )
    direction = d.load_direction(direction_json)
    assert direction.held_out_seed_proj is not None

    result = calibrate_threshold_from_direction(direction, target_precision=0.8)
    assert "exact empirical precision-recall curve" in result["method"]

    # Highest-qualifying-z contract: exactly z=5.0 under the fixed semantics.
    assert result["threshold"] == 5.0


def test_calibrate_threshold_falls_back_to_gaussian_when_no_raw_projections(tmp_path, scene):
    """Artifacts without the raw arrays (pre-#66, or empty held_out_auc
    block) keep the honestly-labeled gaussian-quantile approximation."""
    direction = d.load_direction(scene["direction_json"])
    assert direction.held_out_seed_proj is None
    result = calibrate_threshold_from_direction(direction, target_precision=0.9)
    assert "gaussian-quantile approximation" in result["method"]


def test_calibrate_threshold_falls_back_when_target_precision_unreachable(tmp_path):
    """When even the highest-z candidate threshold cannot reach
    target_precision on the held-out split, fall back to the gaussian
    approximation rather than returning a threshold that overclaims."""
    mu = np.zeros(DIM)
    calibration = _make_calibration(mu, "unused.f32", 20)
    unit_axis = np.zeros(DIM)
    unit_axis[0] = 1.0

    # Perfectly interleaved -- no threshold ever reaches 0.99 precision.
    seed_proj = np.array([0.0, 1.0, 2.0])
    neg_proj = np.array([0.0, 1.0, 2.0])

    direction_json = _write_direction_artifact(
        tmp_path, calibration, unit_axis,
        null_mu0=0.0, null_sigma0=1.0, held_out_auc=0.5,
        held_out_seed_proj=seed_proj, held_out_negative_proj=neg_proj,
    )
    direction = d.load_direction(direction_json)
    result = calibrate_threshold_from_direction(direction, target_precision=0.99)
    assert "gaussian-quantile approximation" in result["method"]


# --------------------------------------------------------------------------- #
# refuse_unless_projection_matches_direction: identity gate.
# --------------------------------------------------------------------------- #

def test_refuse_unless_projection_matches_direction_passes_on_match():
    direction = _direction_with_verdict("usable")
    projection = _make_projection([0], [1.0])
    projection.direction_sha256 = direction.direction_sha256
    projection.source_memmap_sha256 = direction.source_memmap_sha256
    refuse_unless_projection_matches_direction(projection, direction)  # no raise


def test_refuse_unless_projection_matches_direction_refuses_on_direction_sha_mismatch():
    direction = _direction_with_verdict("usable")
    projection = _make_projection([0], [1.0])
    projection.source_memmap_sha256 = direction.source_memmap_sha256
    projection.direction_sha256 = "different-sha"
    with pytest.raises(DirectionRefusal, match="direction_sha256"):
        refuse_unless_projection_matches_direction(projection, direction)


def test_refuse_unless_projection_matches_direction_refuses_on_memmap_sha_mismatch():
    direction = _direction_with_verdict("usable")
    projection = _make_projection([0], [1.0])
    projection.direction_sha256 = direction.direction_sha256
    projection.source_memmap_sha256 = "different-memmap-sha"
    with pytest.raises(DirectionRefusal, match="source_memmap_sha256"):
        refuse_unless_projection_matches_direction(projection, direction)


# --------------------------------------------------------------------------- #
# query_rates: MCP handler end-to-end, threshold+precision+shas recorded.
# --------------------------------------------------------------------------- #

def _build_projection_artifact(tmp_path, scene):
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections"),
            },
        )
    )
    assert "error" not in result, result
    return result["projection_ref"]


def test_query_rates_end_to_end_with_direction_calibrated_threshold(tmp_path, scene, corpus_db):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        query_rates(
            manager=None,
            args={
                "projection_ref": projection_ref,
                "corpus_db_ref": corpus_db,
                "direction_ref": scene["direction_json"],
                "out_dir": str(tmp_path / "rates"),
            },
        )
    )
    assert "error" not in result, result
    assert result["threshold"]["source_auc"] == pytest.approx(0.85)
    assert result["rate_table"]["n_total"] == scene["n_used"]
    assert os.path.isfile(result["rate_table_ref"])

    manifest = json.loads(Path(result["rate_table_ref"]).read_text())
    assert manifest["header"]["regime"] == "direction-rate-table"
    assert manifest["threshold"]["threshold"] == result["threshold"]["threshold"]
    assert manifest["header"]["source_memmap_sha256"] == scene["calibration"].source_memmap_sha256
    assert manifest["header"]["direction_sha256"]

    era_keys = {g["key"]["era"] for g in manifest["rate_table"]["groups"]}
    assert "NULL" in era_keys  # NULL era survives into the persisted artifact


def test_query_rates_with_explicit_threshold_skips_direction_calibration(tmp_path, scene, corpus_db):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        query_rates(
            manager=None,
            args={
                "projection_ref": projection_ref,
                "corpus_db_ref": corpus_db,
                "threshold": 5.0,
                "out_dir": str(tmp_path / "rates"),
            },
        )
    )
    assert "error" not in result, result
    assert result["threshold"]["threshold"] == 5.0
    assert result["threshold"]["method"] == "explicit threshold supplied by caller"


def test_query_rates_requires_threshold_or_direction_ref(tmp_path, scene, corpus_db):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        query_rates(
            manager=None,
            args={"projection_ref": projection_ref, "corpus_db_ref": corpus_db},
        )
    )
    assert "error" in result
    assert "threshold" in result["error"] or "direction_ref" in result["error"]


def test_query_rates_refuses_on_projection_direction_mismatch(tmp_path, scene, corpus_db):
    """A projection artifact built from one direction must refuse
    query_rates against a DIFFERENT direction artifact -- even one with the
    same pattern_id/era slug, since np.savez output for identical array
    content is byte-identical regardless of filename (verified separately),
    so the mismatch must come from genuinely different unit_axis content,
    not merely a different pattern_id string."""
    projection_ref = _build_projection_artifact(tmp_path, scene)
    different_axis = np.zeros(DIM)
    different_axis[-1] = 1.0  # orthogonal to scene["true_axis"] (dim 0)
    other_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], different_axis, pattern_id="unrelated-pattern",
    )
    result = _run(
        query_rates(
            manager=None,
            args={
                "projection_ref": projection_ref,
                "corpus_db_ref": corpus_db,
                "direction_ref": other_direction,
                "out_dir": str(tmp_path / "rates"),
            },
        )
    )
    assert "error" in result
    assert "direction_sha256" in result["error"]


def test_query_rates_requires_refs():
    result = _run(query_rates(manager=None, args={}))
    assert "error" in result
    assert "projection_ref" in result["error"]


# --------------------------------------------------------------------------- #
# top_exemplars: ranked chunk_ids + text readback via corpus.db read-only.
# --------------------------------------------------------------------------- #

def test_top_exemplars_ranks_by_descending_z_and_fetches_text(tmp_path, scene, corpus_db):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        top_exemplars(
            manager=None,
            args={"projection_ref": projection_ref, "corpus_db_ref": corpus_db, "k": 3},
        )
    )
    assert "error" not in result, result
    assert len(result["exemplars"]) == 3
    zs = [e["z"] for e in result["exemplars"]]
    assert zs == sorted(zs, reverse=True)
    # Highest-displacement rows are 19, 18, 17 (scene's monotonic axis).
    assert result["exemplars"][0]["rowid_mm"] == 19
    assert result["exemplars"][0]["chunk_id"] == "chunk-19"
    assert result["exemplars"][0]["text"] == "text body 19"


def test_top_exemplars_era_filter_restricts_ranking(tmp_path, scene, corpus_db):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        top_exemplars(
            manager=None,
            args={
                "projection_ref": projection_ref,
                "corpus_db_ref": corpus_db,
                "k": 5,
                "era": "Gemini",
            },
        )
    )
    assert "error" not in result, result
    assert len(result["exemplars"]) > 0
    conn = sqlite3.connect(corpus_db)
    for ex in result["exemplars"]:
        row = conn.execute(
            "SELECT era FROM chunk WHERE chunk_id = ?", (ex["chunk_id"],)
        ).fetchone()
        assert row[0] == "Gemini"
    conn.close()


def test_top_exemplars_writes_nothing_to_corpus_db(tmp_path, scene, corpus_db):
    """Read-only assertion (D6 boundary discipline, applied here too): a
    corpus.db mtime/content check before and after confirms top_exemplars
    performs no write."""
    projection_ref = _build_projection_artifact(tmp_path, scene)
    before = Path(corpus_db).read_bytes()
    _run(
        top_exemplars(
            manager=None,
            args={"projection_ref": projection_ref, "corpus_db_ref": corpus_db, "k": 5},
        )
    )
    after = Path(corpus_db).read_bytes()
    assert before == after


def test_top_exemplars_requires_refs():
    result = _run(top_exemplars(manager=None, args={}))
    assert "error" in result
    assert "projection_ref" in result["error"]


# --------------------------------------------------------------------------- #
# Projection artifact loader: hard-fail discipline (mirrors load_direction).
# --------------------------------------------------------------------------- #

def test_load_projection_missing_file_hard_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_projection(str(tmp_path / "nope.proj.json"))


def test_load_projection_headerless_refused(tmp_path):
    path = tmp_path / "x.proj.json"
    path.write_text(json.dumps({"n_rows": 0}))
    with pytest.raises(ValueError, match="header"):
        load_projection(str(path))


@pytest.mark.parametrize(
    "missing_key",
    ["regime", "embedding_model_id", "source_memmap_sha256", "direction_sha256", "pattern_id", "era", "n_used"],
)
def test_load_projection_missing_required_header_key_refused(tmp_path, missing_key):
    header = {
        "regime": "direction-projection",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "direction_sha256": "deadbeef",
        "pattern_id": "p",
        "era": None,
        "n_used": 10,
    }
    del header[missing_key]
    path = tmp_path / "x.proj.json"
    path.write_text(json.dumps({"header": header}))
    with pytest.raises(ValueError, match="missing required keys"):
        load_projection(str(path))


def test_load_projection_wrong_regime_refused(tmp_path):
    header = {
        "regime": "functional-direction",  # wrong regime
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "direction_sha256": "deadbeef",
        "pattern_id": "p",
        "era": None,
        "n_used": 10,
    }
    path = tmp_path / "x.proj.json"
    path.write_text(json.dumps({"header": header}))
    with pytest.raises(ValueError, match="regime"):
        load_projection(str(path))


def test_load_projection_missing_npz_hard_fails(tmp_path):
    header = {
        "regime": "direction-projection",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "direction_sha256": "deadbeef",
        "pattern_id": "p",
        "era": None,
        "n_used": 10,
    }
    path = tmp_path / "p.proj.json"
    path.write_text(json.dumps({"header": header}))
    # No sibling .npz written.
    with pytest.raises(FileNotFoundError, match="npz"):
        load_projection(str(path))


def test_load_projection_missing_array_key_refused(tmp_path):
    header = {
        "regime": "direction-projection",
        "embedding_model_id": EMBEDDING_MODEL_ID,
        "source_memmap_sha256": MEMMAP_SHA,
        "direction_sha256": "deadbeef",
        "pattern_id": "p",
        "era": None,
        "n_used": 10,
    }
    json_path = tmp_path / "p.proj.json"
    json_path.write_text(json.dumps({"header": header}))
    npz_path = tmp_path / "p.proj.npz"
    np.savez(npz_path, rowid_mm=np.array([0, 1]))  # missing 'z'
    with pytest.raises(ValueError, match="'z'"):
        load_projection(str(json_path))


def test_load_projection_round_trip(tmp_path, scene):
    npz_path, json_path = write_projection_artifact(
        out_dir=str(tmp_path / "projections"),
        direction=d.load_direction(scene["direction_json"]),
        rowids_mm=np.array([0, 1, 2]),
        projection=np.array([1.0, 2.0, 3.0]),
        z=np.array([0.1, 0.2, 0.3]),
        n_used=scene["n_used"],
    )
    loaded = load_projection(json_path)
    np.testing.assert_allclose(loaded.rowid_mm, [0, 1, 2])
    np.testing.assert_allclose(loaded.z, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(loaded.projection, [1.0, 2.0, 3.0])
    assert loaded.n_used == scene["n_used"]


# --------------------------------------------------------------------------- #
# server.py registration: complete-surface (ARCHITECTURE Rule 2), text-scan
# per the disclosed sk-mcp#63 SDK-drift limitation (mirrors
# test_direction_extraction.py's precedent -- import avoided entirely).
# --------------------------------------------------------------------------- #

_SERVER_PY_PATH = (
    Path(__file__).resolve().parent.parent
    / "semantic_kinematics"
    / "mcp"
    / "server.py"
)


def _read_server_source() -> str:
    return _SERVER_PY_PATH.read_text(encoding="utf-8")


def test_phase3_tools_all_have_a_call_tool_dispatch_arm():
    server_src = _read_server_source()
    phase3_names = {"project_text", "project_chunks", "project_corpus", "query_rates", "top_exemplars"}
    declared = {tool.name for tool in d.get_tools()}
    assert phase3_names <= declared
    for name in phase3_names:
        assert f'name == "{name}"' in server_src, f"tool {name!r} has no call_tool dispatch arm in server.py"


def test_phase3_tools_dispatch_to_direction_module():
    server_src = _read_server_source()
    for name in ["project_text", "project_chunks", "project_corpus", "query_rates", "top_exemplars"]:
        assert f"direction.{name}(state_manager, arguments)" in server_src


# --------------------------------------------------------------------------- #
# Statelessness (ARCHITECTURE Rule 1): same-in/same-out.
# --------------------------------------------------------------------------- #

def test_project_corpus_is_deterministic_same_in_same_out(tmp_path, scene):
    result1 = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "run1"),
            },
        )
    )
    result2 = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": scene["direction_json"],
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "run2"),
            },
        )
    )
    loaded1 = load_projection(result1["projection_ref"])
    loaded2 = load_projection(result2["projection_ref"])
    np.testing.assert_allclose(loaded1.z, loaded2.z)
    np.testing.assert_array_equal(loaded1.rowid_mm, loaded2.rowid_mm)


def test_project_chunks_is_deterministic_same_in_same_out(tmp_path, scene):
    args = {
        "direction_ref": scene["direction_json"],
        "calibration_ref": scene["calibration_json"],
        "rowids": [1, 2, 3],
    }
    result1 = _run(project_chunks(manager=None, args=dict(args)))
    result2 = _run(project_chunks(manager=None, args=dict(args)))
    assert result1["rows"] == result2["rows"]


def test_build_rate_table_is_pure_no_module_level_state():
    """No module-level mutable state backs build_rate_table -- two
    independent calls over identical inputs give identical output (a proxy
    for 'no cross-call retained state' since the function takes no manager)."""
    projection = _make_projection([0, 1], [1.0, 2.0])
    corpus_rows = [
        {"rowid_mm": 0, "era": "Opus 4.8", "channel": "a", "speaker": "assistant"},
        {"rowid_mm": 1, "era": "Opus 4.8", "channel": "a", "speaker": "assistant"},
    ]
    table1 = build_rate_table(projection, 1.5, corpus_rows)
    table2 = build_rate_table(projection, 1.5, corpus_rows)
    assert table1 == table2


# --------------------------------------------------------------------------- #
# Coverage-completing: required-arg guards, load-failure branches,
# _standard_normal_quantile's tail branches, empty-input helper branches.
# --------------------------------------------------------------------------- #

def test_standard_normal_quantile_rejects_out_of_domain():
    with pytest.raises(ValueError, match="0 < p < 1"):
        d._standard_normal_quantile(0.0)
    with pytest.raises(ValueError, match="0 < p < 1"):
        d._standard_normal_quantile(1.0)


def test_standard_normal_quantile_low_tail_branch():
    # p < 0.02425 takes the low-tail rational-approximation branch.
    q = d._standard_normal_quantile(0.001)
    assert q < -3.0


def test_standard_normal_quantile_high_tail_matches_low_tail_by_symmetry():
    low = d._standard_normal_quantile(0.001)
    high = d._standard_normal_quantile(0.999)
    assert high == pytest.approx(-low, abs=1e-6)


def test_fetch_chunk_text_empty_chunk_ids_returns_empty(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        assert fetch_chunk_text(conn, []) == {}
    finally:
        conn.close()


def test_fetch_chunk_id_for_rowid_empty_rowids_returns_empty(corpus_db):
    conn = open_corpus_db_readonly(corpus_db)
    try:
        assert fetch_chunk_id_for_rowid(conn, []) == {}
    finally:
        conn.close()


def test_resolve_memmap_path_relative_resolves_against_cwd_when_present(tmp_path, monkeypatch):
    (tmp_path / "vectors.f32").write_bytes(b"\x00" * 16)
    monkeypatch.chdir(tmp_path)
    resolved = resolve_memmap_path("vectors.f32", None)
    assert os.path.isfile(resolved)


def test_project_text_requires_direction_ref():
    result = _run(project_text(manager=None, args={"text": "x", "calibration_ref": "y"}))
    assert "error" in result
    assert "direction_ref" in result["error"]


def test_project_text_requires_calibration_ref():
    result = _run(project_text(manager=None, args={"text": "x", "direction_ref": "y"}))
    assert "error" in result
    assert "calibration_ref" in result["error"]


def test_project_text_z_scores_zero_sigma_reports_z_error_not_crash(tmp_path, scene):
    """A direction whose recorded null_reference has sigma0=0 (degenerate)
    must not crash project_text; z_scores' DirectionRefusal is caught and
    surfaced as a 'z_error' field, with projection/cosine still returned."""
    zero_sigma_direction = _write_direction_artifact(
        tmp_path, scene["calibration"], scene["true_axis"], pattern_id="zero-sigma",
        null_sigma0=0.0,
    )
    adapter = _FakeAdapter(EMBEDDING_MODEL_ID, {"x": 3.0})
    manager = _FakeManager(adapter)
    result = _run(
        project_text(
            manager=manager,
            args={"text": "x", "direction_ref": zero_sigma_direction, "calibration_ref": scene["calibration_json"]},
        )
    )
    assert "error" not in result
    assert "z_error" in result
    assert "sigma0" in result["z_error"]
    assert "projection" in result


def test_project_chunks_requires_direction_ref():
    result = _run(project_chunks(manager=None, args={"calibration_ref": "y", "rowids": [0]}))
    assert "error" in result
    assert "direction_ref" in result["error"]


def test_project_chunks_requires_calibration_ref():
    result = _run(project_chunks(manager=None, args={"direction_ref": "x", "rowids": [0]}))
    assert "error" in result
    assert "calibration_ref" in result["error"]


def test_project_corpus_requires_calibration_ref():
    result = _run(project_corpus(manager=None, args={"direction_ref": "x"}))
    assert "error" in result
    assert "calibration_ref" in result["error"]


def test_query_rates_requires_corpus_db_ref():
    result = _run(query_rates(manager=None, args={"projection_ref": "x"}))
    assert "error" in result
    assert "corpus_db_ref" in result["error"]


def test_query_rates_handler_reports_load_projection_failure(tmp_path, corpus_db):
    result = _run(
        query_rates(
            manager=None,
            args={"projection_ref": str(tmp_path / "nope.proj.json"), "corpus_db_ref": corpus_db, "threshold": 1.0},
        )
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_query_rates_handler_reports_corpus_db_open_failure(tmp_path, scene):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        query_rates(
            manager=None,
            args={"projection_ref": projection_ref, "corpus_db_ref": str(tmp_path / "nope.db"), "threshold": 1.0},
        )
    )
    assert "error" in result
    assert "corpus.db not found" in result["error"]


def test_top_exemplars_requires_corpus_db_ref():
    result = _run(top_exemplars(manager=None, args={"projection_ref": "x"}))
    assert "error" in result
    assert "corpus_db_ref" in result["error"]


def test_top_exemplars_handler_reports_load_projection_failure(tmp_path, corpus_db):
    result = _run(
        top_exemplars(
            manager=None,
            args={"projection_ref": str(tmp_path / "nope.proj.json"), "corpus_db_ref": corpus_db},
        )
    )
    assert "error" in result
    assert "not found" in result["error"]


def test_top_exemplars_handler_reports_corpus_db_open_failure(tmp_path, scene):
    projection_ref = _build_projection_artifact(tmp_path, scene)
    result = _run(
        top_exemplars(
            manager=None,
            args={"projection_ref": projection_ref, "corpus_db_ref": str(tmp_path / "nope.db")},
        )
    )
    assert "error" in result
    assert "corpus.db not found" in result["error"]


def test_top_exemplars_skips_rowid_with_no_chunk_id_mapping(tmp_path, scene):
    """A projection artifact can cover a rowid_mm that corpus.db's
    vector_status no longer maps to a chunk_id (e.g. a since-pruned row);
    top_exemplars skips it rather than crashing on a None chunk_id."""
    npz_path, json_path = write_projection_artifact(
        out_dir=str(tmp_path / "projections"),
        direction=d.load_direction(scene["direction_json"]),
        rowids_mm=np.array([0, 99999]),  # 99999 has no corpus.db row
        projection=np.array([1.0, 100.0]),
        z=np.array([1.0, 100.0]),  # rank it first if it were included
        n_used=scene["n_used"],
    )
    rows = [
        {
            "rowid_mm": 0,
            "chunk_id": "chunk-0",
            "era": "Opus 4.8",
            "source": "claude_code",
            "speaker": "assistant",
            "text": "text body 0",
        }
    ]
    partial_db_dir = tmp_path / "partial_db_dir"
    partial_db_dir.mkdir()
    corpus_db_path = _make_corpus_db(partial_db_dir, rows)
    result = _run(
        top_exemplars(
            manager=None,
            args={"projection_ref": json_path, "corpus_db_ref": corpus_db_path, "k": 5},
        )
    )
    assert "error" not in result, result
    assert len(result["exemplars"]) == 1
    assert result["exemplars"][0]["chunk_id"] == "chunk-0"


# --------------------------------------------------------------------------- #
# Z-score vs raw-projection unit contract (PR #65 review BLOCKER regression).
#
# The defect this class enforces against: project_corpus persisted the RAW
# projection under the 'z' key, and build_rate_table/top_exemplars then
# thresholded/ranked those raw values against calibrate_threshold_from_
# direction's standard-normal-quantile output -- a unit mismatch invisible
# whenever the null_reference happened to be mu0=0,sigma0=1 (as every prior
# fixture was). These tests use a non-trivial null (mu0=0.3, sigma0=2.5) so
# the z-scale and the raw-projection scale genuinely diverge; absence of this
# class is exactly what masked the defect.
# --------------------------------------------------------------------------- #

_NULL_MU0 = 0.3
_NULL_SIGMA0 = 2.5


def _project_corpus_with_nontrivial_null(tmp_path, scene):
    """Build a direction whose recorded null_reference is mu0=0.3, sigma0=2.5,
    project the scene corpus through it, and return the loaded artifact. The
    scene's axis is dim-0-aligned with mu=0, so the raw projection of row i is
    exactly its displacement (i)."""
    direction_json = _write_direction_artifact(
        tmp_path,
        scene["calibration"],
        scene["true_axis"],
        pattern_id="nontrivial-null",
        null_mu0=_NULL_MU0,
        null_sigma0=_NULL_SIGMA0,
    )
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": direction_json,
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections_nn"),
            },
        )
    )
    assert "error" not in result, result
    return direction_json, load_projection(result["projection_ref"])


def test_project_corpus_persists_genuine_zscore_distinct_from_raw_projection(tmp_path, scene):
    """(a) With sigma0 != 1 the persisted z is a genuine z-score, NOT the raw
    projection: z_i == (raw_i - mu0)/sigma0, and the two arrays differ."""
    _direction_json, loaded = _project_corpus_with_nontrivial_null(tmp_path, scene)

    # Raw projection of row i is its displacement i (dim-0 axis, mu=0).
    raw_by_rowid = dict(zip(loaded.rowid_mm.tolist(), loaded.projection.tolist()))
    z_by_rowid = dict(zip(loaded.rowid_mm.tolist(), loaded.z.tolist()))
    for i, displacement in enumerate(scene["displacements"]):
        assert raw_by_rowid[i] == pytest.approx(displacement, abs=1e-4)
        expected_z = (displacement - _NULL_MU0) / _NULL_SIGMA0
        assert z_by_rowid[i] == pytest.approx(expected_z, abs=1e-4)

    # The two arrays are genuinely different units (the defect stored raw as z).
    assert not np.allclose(loaded.projection, loaded.z)
    # And the z matches z_scores() applied to the raw projection.
    np.testing.assert_allclose(
        loaded.z, z_scores(loaded.projection, _NULL_MU0, _NULL_SIGMA0), atol=1e-9
    )


def test_rate_table_at_zscale_threshold_differs_from_raw_projection_interpretation(tmp_path, scene):
    """(b) A rate table computed at a z-scale threshold changes vs. the
    raw-projection interpretation. At z-threshold t, a chunk counts as 'above'
    iff (raw - mu0)/sigma0 >= t, i.e. raw >= mu0 + t*sigma0 -- which is a
    DIFFERENT raw cutoff than treating t as a raw threshold. If the artifact
    had (buggy) raw values under 'z', the same threshold would select a
    different set, changing the rate."""
    _direction_json, loaded = _project_corpus_with_nontrivial_null(tmp_path, scene)

    corpus_rows = [
        {"rowid_mm": i, "era": "Opus 4.8", "channel": "claude_code", "speaker": "assistant"}
        for i in range(scene["n_used"])
    ]

    z_threshold = 2.0  # a standard-normal-quantile-scale threshold
    # On the genuine z-scale: raw >= mu0 + z*sigma0 = 0.3 + 2.0*2.5 = 5.3, so
    # displacements 6..19 count above -> 14 of 20.
    table_z = build_rate_table(loaded, threshold=z_threshold, corpus_rows=corpus_rows)
    assert table_z["n_total"] == scene["n_used"]
    n_above_z = table_z["groups"][0]["n_above"]
    assert n_above_z == 14  # displacements >= 5.3

    # The raw-projection interpretation of the SAME numeric threshold (the
    # buggy path): raw >= 2.0 -> displacements 2..19 count -> 18 of 20.
    raw_projection_fake = _make_projection(
        loaded.rowid_mm.tolist(), loaded.projection.tolist()
    )
    table_raw = build_rate_table(
        raw_projection_fake, threshold=z_threshold, corpus_rows=corpus_rows
    )
    n_above_raw = table_raw["groups"][0]["n_above"]
    assert n_above_raw == 18  # displacements >= 2.0

    # The two interpretations disagree -- the unit mismatch is observable.
    assert n_above_z != n_above_raw
    assert table_z["groups"][0]["rate"] != table_raw["groups"][0]["rate"]


def test_project_corpus_refuses_zero_sigma_null_reference(tmp_path, scene):
    """project_corpus handles the zero-sigma refusal consistently with
    z_scores' existing behavior: a degenerate null_reference (sigma0=0) is
    refused with the sigma0 message, not persisted as a raw-as-z artifact."""
    zero_sigma_direction = _write_direction_artifact(
        tmp_path,
        scene["calibration"],
        scene["true_axis"],
        pattern_id="zero-sigma-corpus",
        null_sigma0=0.0,
    )
    result = _run(
        project_corpus(
            manager=None,
            args={
                "direction_ref": zero_sigma_direction,
                "calibration_ref": scene["calibration_json"],
                "out_dir": str(tmp_path / "projections_zero"),
            },
        )
    )
    assert "error" in result
    assert "sigma0" in result["error"]
