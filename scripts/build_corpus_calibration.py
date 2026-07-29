#!/usr/bin/env python3
"""Build the corpus-calibration artifact (ADR-SKM-008 §D1, Phase 1).

Reads the TVI-008 frozen boundary contract -- the dense float32 memmap
(``vectors_nv-embed-v2.f32``) plus its ``corpus_join_manifest.json`` sidecar --
and emits a self-describing, model-keyed calibration artifact:

    data/calibration/<embedding_model_slug>.calibration.npz   # mu, [eigvecs, eigvals]
    data/calibration/<embedding_model_slug>.calibration.json  # manifest

The manifest is regime-typed (``regime: corpus-calibration``) and follows the
jolt.py ``load_null`` hard-fail discipline: this script REFUSES loudly (no
trust-and-degrade) when the tvi manifest is missing a required key, declares
an unrecognized ``convention_version``, or when the memmap's recomputed
sha256 does not match the tvi manifest's declared ``source_memmap_sha256``.
There is no silent fallback to a stale or mismatched calibration.

The numeric core (mu, uncentered/centered eigenspectra) is lifted from
``scripts/measure_cone.py`` (see the anisotropy-instruments runbook), factored
here into pure, IO-free functions over an already-loaded ``(N, d)`` array so it
is testable without a real corpus.

Usage:
    python scripts/build_corpus_calibration.py \\
        --tvi-manifest /path/to/corpus_join_manifest.json \\
        --eigenbasis-k 256

    # dry structural check against a small synthetic manifest+memmap pair
    python scripts/build_corpus_calibration.py \\
        --tvi-manifest tests/fixtures/tiny_manifest.json --out-dir /tmp/cal

Exit codes: 2 = tvi manifest/memmap not found or unreadable, 3 = manifest
missing a required key, 4 = unrecognized convention_version, 5 = memmap sha256
mismatch (refuse-on-mismatch, ADR-001 lineage). Structured JSON result goes to
stdout; progress goes to stderr (house style per build_axis_null.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The one convention this script understands. A tvi manifest declaring any
# other value is refused -- no legacy reader, no silent degrade (see
# ADR-SKM-008 D1 and the ADR-001 null-cache refuse-on-mismatch lineage).
EXPECTED_CONVENTION_VERSION = "tvi-008-dedup-keep-last-zerofilter-v1"
EXPECTED_REGIME = "corpus-calibration"

# Required keys on the tvi corpus_join_manifest.json -- the frozen boundary
# contract this script reads (thought-vault-integration#51 / ADR-TVI-008).
REQUIRED_TVI_MANIFEST_KEYS = (
    "source_memmap_path",
    "source_memmap_sha256",
    "n_used",
    "convention_version",
    "embedding_model_id",
    "dimensions",
)

DEFAULT_EIGENBASIS_K = 256
CUM_K = (1, 5, 10, 34, 50, 100, 256)

OUTPUT_DIR = _REPO_ROOT / "data" / "calibration"


class CalibrationRefusal(Exception):
    """Raised for every REFUSE case (missing key / sha mismatch / bad convention).

    Carries an ``exit_code`` so main() can propagate a loud, typed exit.
    """

    def __init__(self, message: str, exit_code: int):
        super().__init__(message)
        self.exit_code = exit_code


# --------------------------------------------------------------------------- #
# tvi manifest loading + refusal gates (no trust-and-degrade).
# --------------------------------------------------------------------------- #

def load_tvi_manifest(path: str) -> Dict[str, Any]:
    """Load and validate the tvi ``corpus_join_manifest.json`` sidecar.

    Refuses (raises :class:`CalibrationRefusal`) on a missing file, a missing
    required key, or an unrecognized ``convention_version``. This is the
    jolt.py ``load_null`` pattern applied to the tvi boundary contract.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError as exc:
        raise CalibrationRefusal(
            f"tvi manifest not found: {path}. Build TVI-008 Phase 1 first "
            "(no silent default).",
            exit_code=2,
        ) from exc
    except json.JSONDecodeError as exc:
        raise CalibrationRefusal(
            f"tvi manifest at {path} is not valid JSON: {exc}", exit_code=2
        ) from exc

    missing = [k for k in REQUIRED_TVI_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise CalibrationRefusal(
            f"tvi manifest {path} missing required keys {missing}; refusing "
            "to calibrate against an under-described boundary contract.",
            exit_code=3,
        )

    if manifest["convention_version"] != EXPECTED_CONVENTION_VERSION:
        raise CalibrationRefusal(
            f"tvi manifest {path} convention_version is "
            f"{manifest['convention_version']!r}, expected "
            f"{EXPECTED_CONVENTION_VERSION!r}. Unrecognized dedup/zero-filter "
            "convention -- refusing to calibrate against it.",
            exit_code=4,
        )

    return manifest


def sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Stream sha256 of a (potentially multi-GB) file without loading it whole."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def resolve_relative(base_manifest_path: str, maybe_relative: str) -> str:
    """Resolve a tvi manifest path entry relative to the tvi repo root.

    tvi's ``corpus_join_manifest.json`` records paths relative to the tvi repo
    root (e.g. ``output/vectors/join/vectors_nv-embed-v2.f32``), not relative
    to the manifest's own directory. Walk up from the manifest's directory
    (``output/vectors/join/``) three levels to the tvi repo root, matching the
    manifest's own path convention. Absolute paths pass through unchanged.
    """
    if os.path.isabs(maybe_relative):
        return maybe_relative
    manifest_dir = Path(base_manifest_path).resolve().parent
    # manifest lives at <tvi-root>/output/vectors/join/corpus_join_manifest.json
    tvi_root = manifest_dir.parent.parent.parent
    candidate = tvi_root / maybe_relative
    if candidate.exists():
        return str(candidate)
    # Fallback: resolve relative to the manifest's own directory (covers
    # synthetic test fixtures that do not mirror the real tvi tree depth).
    return str((manifest_dir / maybe_relative).resolve())


def verify_memmap_sha(memmap_path: str, declared_sha256: str) -> str:
    """Recompute the memmap's sha256 and refuse loudly on a mismatch.

    Returns the recomputed (verified) sha256 on success. This is the
    identity gate every downstream artifact inherits (ADR-SKM-008 D1).
    """
    if not os.path.isfile(memmap_path):
        raise CalibrationRefusal(
            f"source memmap not found: {memmap_path}", exit_code=2
        )
    recomputed = sha256_of_file(memmap_path)
    if recomputed != declared_sha256:
        raise CalibrationRefusal(
            f"memmap sha256 mismatch: recomputed {recomputed} != manifest-declared "
            f"{declared_sha256} for {memmap_path}. The memmap has changed since the "
            "tvi manifest was written -- refusing to calibrate against a "
            "since-mutated corpus (ADR-001 refuse-on-mismatch lineage).",
            exit_code=5,
        )
    return recomputed


def load_memmap_matrix(memmap_path: str, n_used: int, dim: int) -> np.ndarray:
    """Open the frozen dense float32 memmap as an (n_used, dim) array.

    The memmap is the already-deduped, already-zero-filtered contract (every
    row is a valid unit-normalized vector) -- no further filtering happens
    here, unlike the JSONL-streaming path in measure_cone.py.
    """
    expected_bytes = n_used * dim * 4
    actual_bytes = os.path.getsize(memmap_path)
    if actual_bytes != expected_bytes:
        raise CalibrationRefusal(
            f"memmap {memmap_path} size {actual_bytes} bytes does not match "
            f"n_used*dim*4 = {expected_bytes} bytes (n_used={n_used}, dim={dim}); "
            "refusing to reshape a memmap that disagrees with its own manifest.",
            exit_code=5,
        )
    return np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n_used, dim))


# --------------------------------------------------------------------------- #
# Pure numeric core (lifted from scripts/measure_cone.py:176-211 -- reused,
# not reinvented). No IO; testable with small hand-built matrices.
# --------------------------------------------------------------------------- #

def compute_mu(x: np.ndarray) -> np.ndarray:
    """Corpus mean vector, computed in float64 regardless of input dtype."""
    return np.asarray(x, dtype=np.float64).mean(axis=0)


def centered_eigenbasis(
    x: np.ndarray, mu: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[int, float]]:
    """Top-k eigenbasis of the CENTERED covariance C = Xc^T Xc / N.

    Mirrors measure_cone.py's centered-spectrum computation (one
    ``np.linalg.eigh`` over the d x d matrix, never a full SVD of X). Returns
    (eigvecs_topk (d, k), eigvals_topk (k,), all_eigvals_desc (d,),
    cumulative_variance_fraction) where the last two are used for calibration
    reporting/`k` justification (ADR-SKM-008 Open Question).

    Eigenvectors are returned as unit-norm columns; ``np.linalg.eigh`` already
    guarantees this and orthonormality among themselves (symmetric matrix).
    """
    xf64 = np.asarray(x, dtype=np.float64)
    n = xf64.shape[0]
    xc = xf64 - mu
    c = (xc.T @ xc) / n
    eigvals_asc, eigvecs_asc = np.linalg.eigh(c)
    # eigh returns ascending order; flip to descending (largest variance first).
    eigvals_desc = eigvals_asc[::-1]
    eigvecs_desc = eigvecs_asc[:, ::-1]

    lam = np.clip(eigvals_desc, 0.0, None)  # guard tiny negative numerical noise
    total = float(np.sum(lam))
    cum = np.cumsum(lam)
    cum_frac = {
        kk: float(cum[min(kk, lam.shape[0]) - 1] / total) if total > 0 else float("nan")
        for kk in CUM_K
    }

    k_eff = min(k, eigvecs_desc.shape[1])
    return eigvecs_desc[:, :k_eff], eigvals_desc[:k_eff], eigvals_desc, cum_frac


def participation_ratio(eigvals_desc: np.ndarray) -> float:
    """Participation ratio PR = (sum lam)^2 / sum(lam^2) over the full spectrum."""
    lam = np.clip(np.asarray(eigvals_desc, dtype=np.float64), 0.0, None)
    total = float(np.sum(lam))
    sumsq = float(np.sum(lam ** 2))
    return (total ** 2) / sumsq if sumsq > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Artifact naming + build orchestration.
# --------------------------------------------------------------------------- #

def model_slug(embedding_model_id: str) -> str:
    """Filesystem-safe slug for an embedding_model_id (e.g. 'nvidia/NV-Embed-v2').

    Replaces '/' with '__' -- reversible, no information loss, no re-typing of
    the canonical id itself (the slug is a filename, never the identity value
    written into the manifest header).
    """
    return embedding_model_id.replace("/", "__")


def build_calibration(
    tvi_manifest_path: str,
    eigenbasis_k: int = DEFAULT_EIGENBASIS_K,
    include_eigenbasis: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run the full calibration build. Returns (npz_arrays, manifest_dict).

    Raises :class:`CalibrationRefusal` on any of the named refuse-on-mismatch
    gates. Progress is NOT printed here (caller's job, per house style); this
    function is the pure/testable orchestration core.
    """
    tvi_manifest = load_tvi_manifest(tvi_manifest_path)

    memmap_path = resolve_relative(tvi_manifest_path, tvi_manifest["source_memmap_path"])
    declared_sha256 = tvi_manifest["source_memmap_sha256"]
    n_used = int(tvi_manifest["n_used"])
    dim = int(tvi_manifest["dimensions"])
    embedding_model_id = tvi_manifest["embedding_model_id"]

    verified_sha256 = verify_memmap_sha(memmap_path, declared_sha256)
    x = load_memmap_matrix(memmap_path, n_used, dim)

    mu = compute_mu(x)
    mu_norm = float(np.linalg.norm(mu))

    npz_arrays: Dict[str, Any] = {"mu": mu.astype(np.float64)}
    eigenbasis_included = False
    eigenbasis_k_used = None
    participation_ratio_centered = None
    cum_frac = None

    if include_eigenbasis:
        eigvecs_topk, eigvals_topk, eigvals_full_desc, cum_frac = centered_eigenbasis(
            x, mu, eigenbasis_k
        )
        npz_arrays["eigvecs"] = eigvecs_topk
        npz_arrays["eigvals"] = eigvals_topk
        eigenbasis_included = True
        eigenbasis_k_used = int(eigvecs_topk.shape[1])
        participation_ratio_centered = participation_ratio(eigvals_full_desc)

    embedding_id_flag = tvi_manifest.get(
        "embedding_id_source",
        "no embedding_id_source field on tvi manifest",
    )

    manifest: Dict[str, Any] = {
        "header": {
            "regime": EXPECTED_REGIME,
            "embedding_model_id": embedding_model_id,
            "embedding_model_id_source": "tvi corpus_join_manifest.json",
            "embedding_model_id_flag": (
                "tvi meta sidecar may carry a re-typed id (tvi#41 class); "
                f"tvi manifest provenance note: {embedding_id_flag}. Canonical "
                "form used here, mismatch surfaced not propagated."
            ),
            "dim": dim,
            "source_memmap_path": tvi_manifest["source_memmap_path"],
            "source_memmap_sha256": verified_sha256,
            "n_used": n_used,
            "convention_version": tvi_manifest["convention_version"],
        },
        "mu_norm": round(mu_norm, 6),
        "eigenbasis_included": eigenbasis_included,
        "eigenbasis_k": eigenbasis_k_used,
        "participation_ratio_centered": (
            round(participation_ratio_centered, 4)
            if participation_ratio_centered is not None
            else None
        ),
        "cumulative_variance_fraction_centered": cum_frac,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    return npz_arrays, manifest


def write_calibration_artifact(
    npz_arrays: Dict[str, Any], manifest: Dict[str, Any], out_dir: str
) -> Tuple[str, str]:
    """Persist the .npz + .json pair, keyed by embedding_model_id slug.

    Returns (npz_path, json_path).
    """
    slug = model_slug(manifest["header"]["embedding_model_id"])
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{slug}.calibration.npz")
    json_path = os.path.join(out_dir, f"{slug}.calibration.json")

    np.savez(npz_path, **npz_arrays)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    return npz_path, json_path


# --------------------------------------------------------------------------- #
# CLI.
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--tvi-manifest",
        required=True,
        help="Path to the tvi corpus_join_manifest.json (the frozen boundary contract).",
    )
    parser.add_argument(
        "--out-dir",
        default=str(OUTPUT_DIR),
        help=f"Output directory for the calibration artifact (default: {OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--eigenbasis-k",
        type=int,
        default=DEFAULT_EIGENBASIS_K,
        help=f"Top-k eigenbasis to persist (default: {DEFAULT_EIGENBASIS_K}, per ADR-SKM-008 D1).",
    )
    parser.add_argument(
        "--no-eigenbasis",
        action="store_true",
        help="Persist mu only, skip the eigh pass (ADR-SKM-008 Option D -- not the default).",
    )
    args = parser.parse_args(argv)

    print(f"[build_corpus_calibration] loading tvi manifest {args.tvi_manifest}", file=sys.stderr)
    try:
        npz_arrays, manifest = build_calibration(
            args.tvi_manifest,
            eigenbasis_k=args.eigenbasis_k,
            include_eigenbasis=not args.no_eigenbasis,
        )
    except CalibrationRefusal as exc:
        print(f"[REFUSE] {exc}", file=sys.stderr)
        return exc.exit_code

    print(
        f"[build_corpus_calibration] mu_norm={manifest['mu_norm']} "
        f"n_used={manifest['header']['n_used']} "
        f"eigenbasis_k={manifest['eigenbasis_k']} "
        f"participation_ratio_centered={manifest['participation_ratio_centered']}",
        file=sys.stderr,
    )

    npz_path, json_path = write_calibration_artifact(npz_arrays, manifest, args.out_dir)
    print(f"[build_corpus_calibration] wrote {npz_path}", file=sys.stderr)
    print(f"[build_corpus_calibration] wrote {json_path}", file=sys.stderr)

    result = {
        "npz_path": npz_path,
        "json_path": json_path,
        "manifest": manifest,
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
