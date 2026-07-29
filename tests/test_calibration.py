"""Tests for the corpus-calibration build script and loader (ADR-SKM-008 Phase 1).

All fixture-based: small synthetic memmap + tvi-manifest pairs written to
tmp_path, NO real corpus, NO network. Covers the acceptance criteria named in
issue #56 / ADR-SKM-008 line 184:

- happy path: build -> load round-trip, mu/eigenbasis numerically sane
- sha-mismatch refusal (a mutated memmap after the manifest was written)
- unrecognized convention_version refusal
- missing required tvi-manifest key refusal
- loader (direction.load_calibration) rejects a header missing a required key
  (embedding_model_id named explicitly, per the acceptance criteria)
- numerical sanity: mu of known vectors, eigenbasis orthonormality, k truncation
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from scripts.build_corpus_calibration import (
    CalibrationRefusal,
    build_calibration,
    centered_eigenbasis,
    compute_mu,
    main as calibration_main,
    model_slug,
    participation_ratio,
    resolve_relative,
    write_calibration_artifact,
)
from semantic_kinematics.mcp.commands.direction import load_calibration


DIM = 8
N = 40


def _make_matrix(rng: np.random.Generator) -> np.ndarray:
    """A small, deterministic, unit-normalized synthetic corpus."""
    x = rng.normal(loc=0.3, scale=1.0, size=(N, DIM)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x


def _write_fixture(tmp_path, rng=None, convention_version="tvi-008-dedup-keep-last-zerofilter-v1"):
    """Write a synthetic memmap + tvi corpus_join_manifest.json pair.

    Returns (manifest_path, memmap_path, X) so tests can assert against the
    exact matrix used to build the fixture.
    """
    rng = rng or np.random.default_rng(0)
    x = _make_matrix(rng)

    memmap_path = tmp_path / "vectors_test.f32"
    x.tofile(memmap_path)

    import hashlib
    sha256 = hashlib.sha256(memmap_path.read_bytes()).hexdigest()

    manifest = {
        "source_memmap_path": str(memmap_path),
        "source_memmap_sha256": sha256,
        "n_used": N,
        "dimensions": DIM,
        "convention_version": convention_version,
        "embedding_model_id": "nvidia/NV-Embed-v2",
        "embedding_id_source": "pinned-constant, pending ADR-TVI-006 Phase 0 mint import",
    }
    manifest_path = tmp_path / "corpus_join_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return str(manifest_path), str(memmap_path), x


# ---------------------------------------------------------------------------
# Happy path: build -> write -> load round-trip
# ---------------------------------------------------------------------------

def test_build_calibration_happy_path(tmp_path):
    manifest_path, _memmap_path, x = _write_fixture(tmp_path)
    npz_arrays, manifest = build_calibration(manifest_path, eigenbasis_k=5)

    assert manifest["header"]["regime"] == "corpus-calibration"
    assert manifest["header"]["embedding_model_id"] == "nvidia/NV-Embed-v2"
    assert manifest["header"]["n_used"] == N
    assert manifest["header"]["dim"] == DIM
    assert manifest["header"]["convention_version"] == "tvi-008-dedup-keep-last-zerofilter-v1"
    assert manifest["eigenbasis_included"] is True
    assert manifest["eigenbasis_k"] == 5

    expected_mu = x.astype(np.float64).mean(axis=0)
    np.testing.assert_allclose(npz_arrays["mu"], expected_mu, rtol=1e-6)
    assert manifest["mu_norm"] == pytest.approx(float(np.linalg.norm(expected_mu)), rel=1e-4)


def test_build_write_load_roundtrip(tmp_path):
    manifest_path, _memmap_path, _x = _write_fixture(tmp_path)
    npz_arrays, manifest = build_calibration(manifest_path, eigenbasis_k=4)
    out_dir = tmp_path / "calibration_out"
    npz_path, json_path = write_calibration_artifact(npz_arrays, manifest, str(out_dir))

    assert os.path.basename(npz_path) == "nvidia__NV-Embed-v2.calibration.npz"
    assert os.path.basename(json_path) == "nvidia__NV-Embed-v2.calibration.json"
    assert os.path.isfile(npz_path)
    assert os.path.isfile(json_path)

    cal = load_calibration(json_path)
    assert cal.embedding_model_id == "nvidia/NV-Embed-v2"
    assert cal.dim == DIM
    assert cal.n_used == N
    assert cal.convention_version == "tvi-008-dedup-keep-last-zerofilter-v1"
    np.testing.assert_allclose(cal.mu, npz_arrays["mu"])
    assert cal.eigvecs.shape == (DIM, 4)
    assert cal.eigvals.shape == (4,)
    assert cal.eigenbasis_k == 4


def test_model_slug_replaces_slash():
    assert model_slug("nvidia/NV-Embed-v2") == "nvidia__NV-Embed-v2"


# ---------------------------------------------------------------------------
# Refusal gates: sha mismatch, bad convention_version, missing required key
# ---------------------------------------------------------------------------

def test_sha_mismatch_refuses(tmp_path):
    manifest_path, memmap_path, _x = _write_fixture(tmp_path)
    # Mutate the memmap after the manifest was written -- the manifest's
    # declared sha256 is now stale.
    with open(memmap_path, "r+b") as fh:
        fh.seek(0)
        fh.write(b"\xff\xff\xff\xff")

    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(manifest_path)
    assert exc_info.value.exit_code == 5
    assert "sha256 mismatch" in str(exc_info.value)


def test_unrecognized_convention_version_refuses(tmp_path):
    manifest_path, _memmap_path, _x = _write_fixture(
        tmp_path, convention_version="some-other-convention-v2"
    )
    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(manifest_path)
    assert exc_info.value.exit_code == 4
    assert "convention_version" in str(exc_info.value)


@pytest.mark.parametrize(
    "missing_key",
    [
        "source_memmap_path",
        "source_memmap_sha256",
        "n_used",
        "convention_version",
        "embedding_model_id",
        "dimensions",
    ],
)
def test_missing_required_tvi_manifest_key_refuses(tmp_path, missing_key):
    manifest_path, _memmap_path, _x = _write_fixture(tmp_path)
    blob = json.loads(Path(manifest_path).read_text())
    del blob[missing_key]
    Path(manifest_path).write_text(json.dumps(blob))

    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(manifest_path)
    assert exc_info.value.exit_code == 3
    assert missing_key in str(exc_info.value)


def test_tvi_manifest_not_found_refuses(tmp_path):
    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(str(tmp_path / "nope.json"))
    assert exc_info.value.exit_code == 2


def test_tvi_manifest_malformed_json_refuses(tmp_path):
    bad_path = tmp_path / "bad_manifest.json"
    bad_path.write_text("{not valid json")
    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(str(bad_path))
    assert exc_info.value.exit_code == 2
    assert "not valid JSON" in str(exc_info.value)


def test_memmap_not_found_refuses(tmp_path):
    manifest_path, memmap_path, _x = _write_fixture(tmp_path)
    os.remove(memmap_path)
    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(manifest_path)
    assert exc_info.value.exit_code == 2


def test_memmap_size_mismatch_refuses(tmp_path):
    """n_used/dim in the manifest disagree with the memmap's actual byte size."""
    manifest_path, _memmap_path, _x = _write_fixture(tmp_path)
    blob = json.loads(Path(manifest_path).read_text())
    blob["n_used"] = N + 1  # now disagrees with the file's real size
    Path(manifest_path).write_text(json.dumps(blob))

    with pytest.raises(CalibrationRefusal) as exc_info:
        build_calibration(manifest_path)
    assert exc_info.value.exit_code == 5
    assert "size" in str(exc_info.value)


# ---------------------------------------------------------------------------
# load_calibration: loader-side refusals (jolt.py load_null discipline)
# ---------------------------------------------------------------------------

def _write_calibration_json(tmp_path, header_overrides=None, top_level_overrides=None, write_npz=True):
    header = {
        "regime": "corpus-calibration",
        "embedding_model_id": "nvidia/NV-Embed-v2",
        "embedding_model_id_source": "tvi corpus_join_manifest.json",
        "dim": 4,
        "source_memmap_path": "vectors.f32",
        "source_memmap_sha256": "abc123",
        "n_used": 10,
        "convention_version": "tvi-008-dedup-keep-last-zerofilter-v1",
    }
    header.update(header_overrides or {})
    manifest = {
        "header": header,
        "mu_norm": 0.5,
        "eigenbasis_included": False,
        "eigenbasis_k": None,
        "built_at": "2026-07-29T00:00:00Z",
    }
    manifest.update(top_level_overrides or {})

    json_path = tmp_path / "nvidia__NV-Embed-v2.calibration.json"
    json_path.write_text(json.dumps(manifest))

    if write_npz:
        npz_path = tmp_path / "nvidia__NV-Embed-v2.calibration.npz"
        np.savez(npz_path, mu=np.zeros(4, dtype=np.float64))

    return str(json_path)


def test_load_calibration_roundtrip(tmp_path):
    json_path = _write_calibration_json(tmp_path)
    cal = load_calibration(json_path)
    assert cal.header["regime"] == "corpus-calibration"
    assert cal.embedding_model_id == "nvidia/NV-Embed-v2"
    assert cal.dim == 4
    assert cal.n_used == 10


def test_load_calibration_missing_file_hard_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_calibration(str(tmp_path / "nope.calibration.json"))


def test_load_calibration_missing_npz_hard_fails(tmp_path):
    json_path = _write_calibration_json(tmp_path, write_npz=False)
    with pytest.raises(FileNotFoundError):
        load_calibration(json_path)


def test_load_calibration_headerless_refused(tmp_path):
    manifest = {"mu_norm": 0.5}
    json_path = tmp_path / "x.calibration.json"
    json_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="header"):
        load_calibration(str(json_path))


def test_load_calibration_missing_embedding_model_id_refused(tmp_path):
    """Acceptance criterion (issue #56): 'loader rejects a header missing
    embedding_model_id'."""
    json_path = _write_calibration_json(tmp_path)
    blob = json.loads(Path(json_path).read_text())
    del blob["header"]["embedding_model_id"]
    Path(json_path).write_text(json.dumps(blob))

    with pytest.raises(ValueError, match="missing required keys") as exc_info:
        load_calibration(json_path)
    # Confirm the specific missing key is named in the error, not just "a key".
    assert "embedding_model_id" in str(exc_info.value)


@pytest.mark.parametrize(
    "missing_key",
    [
        "regime",
        "embedding_model_id",
        "dim",
        "source_memmap_path",
        "source_memmap_sha256",
        "n_used",
        "convention_version",
    ],
)
def test_load_calibration_missing_any_required_key_refused(tmp_path, missing_key):
    json_path = _write_calibration_json(tmp_path)
    blob = json.loads(Path(json_path).read_text())
    del blob["header"][missing_key]
    Path(json_path).write_text(json.dumps(blob))

    with pytest.raises(ValueError, match="missing required keys"):
        load_calibration(json_path)


def test_load_calibration_wrong_regime_refused(tmp_path):
    json_path = _write_calibration_json(tmp_path, header_overrides={"regime": "bearing-magnitude"})
    with pytest.raises(ValueError, match="regime"):
        load_calibration(json_path)


def test_load_calibration_missing_mu_array_refused(tmp_path):
    header = {
        "regime": "corpus-calibration",
        "embedding_model_id": "nvidia/NV-Embed-v2",
        "dim": 4,
        "source_memmap_path": "vectors.f32",
        "source_memmap_sha256": "abc123",
        "n_used": 10,
        "convention_version": "tvi-008-dedup-keep-last-zerofilter-v1",
    }
    manifest = {"header": header, "mu_norm": 0.5}
    json_path = tmp_path / "nvidia__NV-Embed-v2.calibration.json"
    json_path.write_text(json.dumps(manifest))
    npz_path = tmp_path / "nvidia__NV-Embed-v2.calibration.npz"
    np.savez(npz_path, not_mu=np.zeros(4))  # no 'mu' key

    with pytest.raises(ValueError, match="'mu'"):
        load_calibration(str(json_path))


def test_refuse_unless_matches_identity_gate(tmp_path):
    json_path = _write_calibration_json(tmp_path)
    cal = load_calibration(json_path)
    # Matching identity: no raise.
    cal.refuse_unless_matches("nvidia/NV-Embed-v2", "abc123")
    # Mismatched model id.
    with pytest.raises(ValueError, match="embedding_model_id"):
        cal.refuse_unless_matches("some-other-model", "abc123")
    # Mismatched sha.
    with pytest.raises(ValueError, match="source_memmap_sha256"):
        cal.refuse_unless_matches("nvidia/NV-Embed-v2", "deadbeef")


# ---------------------------------------------------------------------------
# Numerical sanity: mu of known vectors, eigenbasis orthonormality, k truncation
# ---------------------------------------------------------------------------

def test_compute_mu_known_vectors():
    x = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    mu = compute_mu(x)
    np.testing.assert_allclose(mu, [0.25, 0.25, 0.25, 0.25])
    assert mu.dtype == np.float64


def test_compute_mu_is_float64_even_for_float32_input():
    x = np.ones((3, 4), dtype=np.float32)
    mu = compute_mu(x)
    assert mu.dtype == np.float64


def test_centered_eigenbasis_orthonormality():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 16))
    mu = compute_mu(x)
    eigvecs, eigvals, eigvals_full, cum_frac = centered_eigenbasis(x, mu, k=6)

    assert eigvecs.shape == (16, 6)
    assert eigvals.shape == (6,)
    assert eigvals_full.shape == (16,)

    # Columns are unit-norm.
    norms = np.linalg.norm(eigvecs, axis=0)
    np.testing.assert_allclose(norms, np.ones(6), atol=1e-8)

    # Columns are mutually orthogonal: V^T V == I.
    gram = eigvecs.T @ eigvecs
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-8)

    # Eigenvalues descending.
    assert np.all(np.diff(eigvals) <= 1e-9)
    assert np.all(np.diff(eigvals_full) <= 1e-9)

    # Cumulative variance fraction is monotonic non-decreasing and <= 1.
    ks = sorted(cum_frac.keys())
    fracs = [cum_frac[k] for k in ks]
    assert all(0.0 <= f <= 1.0 + 1e-9 for f in fracs)
    assert all(fracs[i] <= fracs[i + 1] + 1e-9 for i in range(len(fracs) - 1))


def test_centered_eigenbasis_k_truncation():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(50, 10))
    mu = compute_mu(x)

    eigvecs_3, eigvals_3, _full, _cf = centered_eigenbasis(x, mu, k=3)
    assert eigvecs_3.shape == (10, 3)
    assert eigvals_3.shape == (3,)

    # k larger than dim is clamped to dim, not an error.
    eigvecs_full, eigvals_full, _full2, _cf2 = centered_eigenbasis(x, mu, k=999)
    assert eigvecs_full.shape == (10, 10)
    assert eigvals_full.shape == (10,)


def test_participation_ratio_isotropic_vs_degenerate():
    # Isotropic spectrum (all eigenvalues equal): PR == dim.
    lam_isotropic = np.ones(8)
    assert participation_ratio(lam_isotropic) == pytest.approx(8.0)

    # Fully degenerate (rank-1): PR == 1.
    lam_degenerate = np.array([5.0, 0.0, 0.0, 0.0])
    assert participation_ratio(lam_degenerate) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# resolve_relative: absolute passthrough + tvi-root-relative resolution
# ---------------------------------------------------------------------------

def test_resolve_relative_absolute_passthrough(tmp_path):
    abs_path = str(tmp_path / "vectors.f32")
    assert resolve_relative(str(tmp_path / "manifest.json"), abs_path) == abs_path


def test_resolve_relative_walks_up_to_tvi_root(tmp_path):
    # Mirror the real tvi layout: <tvi-root>/output/vectors/join/manifest.json
    tvi_root = tmp_path / "tvi-root"
    join_dir = tvi_root / "output" / "vectors" / "join"
    join_dir.mkdir(parents=True)
    manifest_path = join_dir / "corpus_join_manifest.json"
    manifest_path.write_text("{}")

    target = tvi_root / "output" / "vectors" / "join" / "vectors.f32"
    target.write_bytes(b"data")

    resolved = resolve_relative(str(manifest_path), "output/vectors/join/vectors.f32")
    assert resolved == str(target)


def test_resolve_relative_falls_back_to_manifest_dir_when_tvi_root_miss(tmp_path):
    # No tvi-root-shaped tree; the candidate file lives next to the manifest
    # instead (covers synthetic fixtures that don't mirror the real depth).
    manifest_path = tmp_path / "corpus_join_manifest.json"
    manifest_path.write_text("{}")
    sibling = tmp_path / "vectors.f32"
    sibling.write_bytes(b"data")

    resolved = resolve_relative(str(manifest_path), "vectors.f32")
    assert resolved == str(sibling.resolve())


# ---------------------------------------------------------------------------
# CLI main(): happy path + refusal exit code, argv-driven
# ---------------------------------------------------------------------------

def test_main_happy_path_exit_zero_and_writes_artifact(tmp_path, capsys):
    manifest_path, _memmap_path, _x = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = calibration_main(
        [
            "--tvi-manifest", manifest_path,
            "--out-dir", str(out_dir),
            "--eigenbasis-k", "3",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["manifest"]["header"]["regime"] == "corpus-calibration"
    assert os.path.isfile(result["npz_path"])
    assert os.path.isfile(result["json_path"])


def test_main_refusal_returns_named_exit_code(tmp_path, capsys):
    manifest_path, _memmap_path, _x = _write_fixture(
        tmp_path, convention_version="unrecognized-convention"
    )
    rc = calibration_main(["--tvi-manifest", manifest_path])
    assert rc == 4
    captured = capsys.readouterr()
    assert "[REFUSE]" in captured.err


def test_main_no_eigenbasis_flag(tmp_path, capsys):
    manifest_path, _memmap_path, _x = _write_fixture(tmp_path)
    out_dir = tmp_path / "out"
    rc = calibration_main(
        ["--tvi-manifest", manifest_path, "--out-dir", str(out_dir), "--no-eigenbasis"]
    )
    assert rc == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["manifest"]["eigenbasis_included"] is False
    assert result["manifest"]["eigenbasis_k"] is None


def test_mu_norm_reconciles_on_synthetic_fixture(tmp_path):
    """Sanity check mirroring the ADR's real-run acceptance criterion (mu_norm
    reconciles to ~0.555 on the real corpus) -- here just asserting the
    build's reported mu_norm matches ||mean(X)|| computed independently."""
    manifest_path, _memmap_path, x = _write_fixture(tmp_path)
    _npz_arrays, manifest = build_calibration(manifest_path, eigenbasis_k=3)
    independent_mu_norm = float(np.linalg.norm(x.astype(np.float64).mean(axis=0)))
    assert manifest["mu_norm"] == pytest.approx(independent_mu_norm, rel=1e-4)
