"""Direction command module: functional-direction probe (ADR-SKM-008).

Phase 1 of ADR-SKM-008 lands only the calibration-artifact loader here --
:func:`load_calibration`. It follows the ``bearing/jolt.py::load_null``
regime-typed self-describing-artifact pattern exactly: hard-fail on a missing
file, a missing/under-described header, or a wrong regime -- no silent
default, no legacy reader.

Later phases (D2-D6: ``initialize_direction``, ``project_*``, ``query_rates``,
``cross_project``, ``direction_diagnostics``, ``preview_pattern``) register
their MCP tools in this module and in ``server.py``; none of that surface
exists yet -- this file currently exposes only the calibration loader Phase 1
needs.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

# Required header keys for a regime-typed calibration artifact (mirrors
# jolt.py's REQUIRED_HEADER_KEYS / EXPECTED_REGIME discipline).
REQUIRED_HEADER_KEYS = (
    "regime",
    "embedding_model_id",
    "dim",
    "source_memmap_path",
    "source_memmap_sha256",
    "n_used",
    "convention_version",
)
EXPECTED_REGIME = "corpus-calibration"


@dataclass
class Calibration:
    """A loaded corpus-calibration artifact (ADR-SKM-008 D1).

    ``mu`` is the float64 corpus mean vector. ``eigvecs``/``eigvals`` are the
    optional top-k centered eigenbasis (``eigvecs`` has shape ``(dim, k)``,
    columns are unit-norm and mutually orthogonal); both are ``None`` when the
    artifact was built with ``--no-eigenbasis``. The eigenbasis is inert until
    used -- a consumer that only mean-centers reads ``mu`` and never touches
    these fields.
    """

    embedding_model_id: str
    dim: int
    source_memmap_path: str
    source_memmap_sha256: str
    n_used: int
    convention_version: str
    mu: np.ndarray
    mu_norm: float
    eigvecs: Optional[np.ndarray]
    eigvals: Optional[np.ndarray]
    eigenbasis_k: Optional[int]
    header: Dict[str, Any]
    manifest: Dict[str, Any]

    def refuse_unless_matches(self, embedding_model_id: str, source_memmap_sha256: str) -> None:
        """Refuse (raise ValueError) if the given identity does not match this
        calibration's. Callers (e.g. a future ``initialize_direction``) use
        this to enforce the calibration<->seedset identity gate (ADR-SKM-008
        D2 step 1) before mean-centering against this artifact's ``mu``.
        """
        mismatches = []
        if embedding_model_id != self.embedding_model_id:
            mismatches.append(
                f"embedding_model_id {embedding_model_id!r} != calibration's "
                f"{self.embedding_model_id!r}"
            )
        if source_memmap_sha256 != self.source_memmap_sha256:
            mismatches.append(
                f"source_memmap_sha256 {source_memmap_sha256!r} != calibration's "
                f"{self.source_memmap_sha256!r}"
            )
        if mismatches:
            raise ValueError(
                "Identity mismatch against calibration artifact: "
                + "; ".join(mismatches)
                + ". A mismatch means the caller indexes a different vector "
                "population than mu was computed over -- refusing rather than "
                "centering against the wrong corpus."
            )


def load_calibration(json_path: str) -> Calibration:
    """Load a corpus-calibration artifact from its self-describing manifest.

    ``json_path`` is the ``<slug>.calibration.json`` manifest written by
    ``scripts/build_corpus_calibration.py``; the sibling ``.npz`` (same
    basename, ``.npz`` extension) is loaded alongside it.

    Hard-fails (no silent default, no legacy reader -- jolt.py ``load_null``
    discipline) on: missing manifest or npz file, a manifest with no 'header'
    block, a header missing a required key, or a header whose ``regime`` is
    not ``corpus-calibration``.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Calibration artifact not found: {json_path}. Build it first "
            "with scripts/build_corpus_calibration.py (no silent default)."
        ) from exc

    header = manifest.get("header")
    if not isinstance(header, dict):
        raise ValueError(
            f"Calibration artifact {json_path} has no 'header' block; "
            "refusing header-less calibration."
        )

    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Calibration artifact {json_path} header missing required keys "
            f"{missing}; refusing to load an under-described calibration."
        )

    if header["regime"] != EXPECTED_REGIME:
        raise ValueError(
            f"Calibration artifact {json_path} regime is {header['regime']!r}, "
            f"expected {EXPECTED_REGIME!r}. Wrong-regime calibration is a "
            "miscalibration."
        )

    # <slug>.calibration.json -> <slug>.calibration.npz (sibling file, same
    # basename minus the final extension).
    npz_path = os.path.splitext(json_path)[0] + ".npz"
    try:
        npz = np.load(npz_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Calibration npz not found: {npz_path} (expected alongside "
            f"{json_path}). Build it first (no silent default)."
        ) from exc

    if "mu" not in npz:
        raise ValueError(
            f"Calibration npz {npz_path} has no 'mu' array; refusing an "
            "artifact without the mean vector every consumer needs."
        )
    mu = np.asarray(npz["mu"], dtype=np.float64)

    eigvecs = np.asarray(npz["eigvecs"]) if "eigvecs" in npz else None
    eigvals = np.asarray(npz["eigvals"]) if "eigvals" in npz else None

    return Calibration(
        embedding_model_id=header["embedding_model_id"],
        dim=int(header["dim"]),
        source_memmap_path=header["source_memmap_path"],
        source_memmap_sha256=header["source_memmap_sha256"],
        n_used=int(header["n_used"]),
        convention_version=header["convention_version"],
        mu=mu,
        mu_norm=float(manifest.get("mu_norm", float(np.linalg.norm(mu)))),
        eigvecs=eigvecs,
        eigvals=eigvals,
        eigenbasis_k=manifest.get("eigenbasis_k"),
        header=header,
        manifest=manifest,
    )
