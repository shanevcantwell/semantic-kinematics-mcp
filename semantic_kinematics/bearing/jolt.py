"""Axis-free jolt detector scored against a measured displacement null.

A jolt is a per-step semantic displacement that is large relative to a baseline
of REAL-TEXT consecutive-step displacements measured on the same embedder at the
same atom (regime ``bearing-magnitude``). There is no projection axis: scoring is
``z = (mag - null_mean) / null_std`` plus the empirical percentile rank of the
step's magnitude against the measured null distribution.

The null is a self-describing artifact (see :func:`load_null`). Loading hard-fails
if the file is missing or its header omits required regime metadata; there is no
silent default and no legacy-format reader (project rule).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Required header keys for a regime-typed null artifact. Atom/embedder/regime are
# load-bearing for atom-matching discipline, not decoration.
REQUIRED_HEADER_KEYS = ("regime", "atom", "embedder", "dim")
EXPECTED_REGIME = "bearing-magnitude"


@dataclass
class DisplacementNull:
    """A measured consecutive-step displacement-magnitude baseline.

    Holds the summary statistics needed to z-score and percentile-rank a new
    step's magnitude, plus the regime header used to enforce atom-matching.
    """

    regime: str
    atom: str
    embedder: str
    dim: int
    n_deltas: int
    mean: float
    std: float
    # Sorted magnitude samples, used for empirical percentile-rank. May be a
    # subsample if the artifact stored one; ``n_deltas`` is the true count.
    sorted_magnitudes: np.ndarray
    percentiles: Dict[str, float]
    header: Dict[str, Any]

    def zscore(self, magnitude: float) -> float:
        """Sigma-above-null for a single magnitude."""
        if self.std <= 0.0:
            # Degenerate null (all deltas identical). Any deviation is infinite
            # sigma; equality is zero. Report honestly rather than dividing by 0.
            return 0.0 if magnitude == self.mean else math.inf
        return (magnitude - self.mean) / self.std

    def percentile_rank(self, magnitude: float) -> float:
        """Empirical percentile rank (0-100) of a magnitude vs the null samples.

        Fraction of null deltas strictly less than ``magnitude``, times 100.
        """
        if self.sorted_magnitudes.size == 0:
            return float("nan")
        idx = int(np.searchsorted(self.sorted_magnitudes, magnitude, side="left"))
        return 100.0 * idx / self.sorted_magnitudes.size


@dataclass
class JoltStep:
    """One trajectory step (transition from embedding i to i+1)."""

    index: int  # step index; displacement from embedding[index] -> embedding[index+1]
    magnitude: float
    z: float
    percentile: float
    label: Optional[str] = None  # text landing on the step (embedding[index+1])


@dataclass
class JoltResult:
    n_steps: int
    threshold_sigma: float
    steps: List[JoltStep]
    flagged: List[JoltStep]  # subset with z >= threshold_sigma
    peak_z: float
    peak_index: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "threshold_sigma": self.threshold_sigma,
            "peak_z": self.peak_z,
            "peak_index": self.peak_index,
            "n_flagged": len(self.flagged),
            "flagged": [
                {
                    "index": s.index,
                    "magnitude": round(s.magnitude, 6),
                    "z": round(s.z, 4),
                    "percentile": round(s.percentile, 4),
                    "label": s.label,
                }
                for s in self.flagged
            ],
        }


def load_null(path: str) -> DisplacementNull:
    """Load a measured displacement null from a self-describing JSON artifact.

    Hard-fails (no silent default) if the file is missing or the header lacks a
    required regime key, or if the regime is not ``bearing-magnitude``.
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Null artifact not found: {path}. Build it first (no silent default)."
        ) from exc

    header = blob.get("header")
    if not isinstance(header, dict):
        raise ValueError(
            f"Null artifact {path} has no 'header' block; refusing header-less null."
        )

    missing = [k for k in REQUIRED_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Null artifact {path} header missing required keys {missing}; "
            f"refusing to load an under-described null."
        )

    if header["regime"] != EXPECTED_REGIME:
        raise ValueError(
            f"Null artifact {path} regime is {header['regime']!r}, "
            f"expected {EXPECTED_REGIME!r}. Wrong-regime null is a miscalibration."
        )

    stats = blob.get("stats")
    if not isinstance(stats, dict) or "mean" not in stats or "std" not in stats:
        raise ValueError(f"Null artifact {path} missing stats.mean/stats.std.")

    samples = blob.get("magnitudes")
    if samples is None:
        raise ValueError(
            f"Null artifact {path} has no 'magnitudes' samples; "
            f"percentile-rank requires the measured distribution."
        )
    sorted_mags = np.sort(np.asarray(samples, dtype=float))

    return DisplacementNull(
        regime=header["regime"],
        atom=header["atom"],
        embedder=header["embedder"],
        dim=int(header["dim"]),
        n_deltas=int(header.get("n_deltas", sorted_mags.size)),
        mean=float(stats["mean"]),
        std=float(stats["std"]),
        sorted_magnitudes=sorted_mags,
        percentiles=stats.get("percentiles", {}),
        header=header,
    )


def displacement_magnitudes(embeddings: np.ndarray) -> np.ndarray:
    """Per-step displacement magnitudes ``||v[i+1] - v[i]||`` for an ordered matrix."""
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D [n_steps, dim], got shape {embeddings.shape}"
        )
    if embeddings.shape[0] < 2:
        return np.array([])
    return np.linalg.norm(np.diff(embeddings, axis=0), axis=1)


def score_jolts(
    embeddings: np.ndarray,
    null: DisplacementNull,
    threshold_sigma: float = 3.0,
    labels: Optional[Sequence[str]] = None,
) -> JoltResult:
    """Score each per-step displacement of an ordered matrix against the null.

    Args:
        embeddings: Ordered embedding matrix ``[n_steps, dim]``.
        null: A loaded measured displacement null (must match the embedder/atom
            of ``embeddings`` -- caller's responsibility; the regime is enforced
            at load time).
        threshold_sigma: Steps with ``z >= threshold_sigma`` are flagged.
        labels: Optional per-EMBEDDING labels (len == n_steps). A step's label is
            the text it lands on, i.e. ``labels[index + 1]``.

    Returns:
        A :class:`JoltResult` with per-step scores and the flagged subset.
    """
    embeddings = np.asarray(embeddings, dtype=float)
    if embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be 2-D [n_steps, dim], got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != null.dim:
        raise ValueError(
            f"embedding dim {embeddings.shape[1]} != null dim {null.dim}; "
            f"refusing to score against a mismatched-dimension null."
        )
    n_steps = embeddings.shape[0]
    if n_steps < 2:
        raise ValueError(f"Need at least 2 steps, got {n_steps}")

    if labels is not None and len(labels) != n_steps:
        raise ValueError(
            f"labels length {len(labels)} != n_steps {n_steps}"
        )

    mags = displacement_magnitudes(embeddings)

    steps: List[JoltStep] = []
    for i, mag in enumerate(mags):
        m = float(mag)
        label = labels[i + 1] if labels is not None else None
        steps.append(
            JoltStep(
                index=i,
                magnitude=m,
                z=null.zscore(m),
                percentile=null.percentile_rank(m),
                label=label,
            )
        )

    flagged = [s for s in steps if s.z >= threshold_sigma]

    peak_idx = int(np.argmax(mags))
    peak_z = steps[peak_idx].z

    return JoltResult(
        n_steps=n_steps,
        threshold_sigma=threshold_sigma,
        steps=steps,
        flagged=flagged,
        peak_z=peak_z,
        peak_index=peak_idx,
    )


# ===========================================================================
# Context-conditioned phrase-displacement null (ADR-SKM-0003).
#
# A separate regime from the sentence ``bearing-magnitude`` null above: the atom
# is a context-conditioned phrase span, and the null is stratified by the step's
# ``(actual_k, length_bucket, demarcator_class)`` because pooling-variance and
# context-overlap are confounded with the comedic signal (ADR Decision 5 /
# Rationale "Why the null is length/k-stratified"). DisplacementNull and
# load_null above are LEFT UNTOUCHED; this is additive, no compat shim.
# ===========================================================================

# Required header keys for the conditioned-null artifact. ``k_range`` and the
# stratum vocabularies are load-bearing for the sparsity-backoff selection.
REQUIRED_CONDITIONED_HEADER_KEYS = (
    "regime",
    "atom",
    "embedder",
    "dim",
    "k_range",
)
EXPECTED_CONDITIONED_REGIME = "bearing-magnitude-conditioned"

# Default minimum stratum size below which scoring backs off to a coarser cell.
DEFAULT_N_MIN = 200


@dataclass
class ConditionedStratum:
    """One ``(k, length, demarcator)`` cell of the conditioned null."""

    mean: float
    std: float
    n: int
    sorted_magnitudes: np.ndarray
    percentiles: Dict[str, float]

    def zscore(self, magnitude: float) -> float:
        if self.std <= 0.0:
            return 0.0 if magnitude == self.mean else math.inf
        return (magnitude - self.mean) / self.std

    def percentile_rank(self, magnitude: float) -> float:
        if self.sorted_magnitudes.size == 0:
            return float("nan")
        idx = int(np.searchsorted(self.sorted_magnitudes, magnitude, side="left"))
        return 100.0 * idx / self.sorted_magnitudes.size


@dataclass
class ConditionedScore:
    """Result of scoring a conditioned step against the stratified null."""

    z: float
    percentile: float
    # Which backoff level actually answered: "k|length|demarcator",
    # "k|length", or "k". Makes the calibration legible (ADR confound #2:
    # if the verdict cell backs off below the demarcator level, the claim
    # drops to a weaker bound -- so the level must be recorded).
    backoff_level: str
    # The concrete stratum key the score was computed against.
    stratum_key: str
    n: int


@dataclass
class ConditionedDisplacementNull:
    """A measured, ``(k, length, demarcator)``-stratified conditioned-phrase null.

    Holds one :class:`ConditionedStratum` per cell plus the header vocabularies
    needed to drive sparsity backoff. The finest key is
    ``"k{K}|{length_bucket}|{demarcator}"``; backoff coarsens to ``"k{K}|{length}"``
    then ``"k{K}"`` when a cell has fewer than ``n_min`` samples.
    """

    regime: str
    atom: str
    embedder: str
    dim: int
    k_range: List[int]
    strata: Dict[str, ConditionedStratum]
    header: Dict[str, Any]

    @staticmethod
    def finest_key(k: int, length_bucket: str, demarcator: str) -> str:
        return f"k{k}|{length_bucket}|{demarcator}"

    @staticmethod
    def length_key(k: int, length_bucket: str) -> str:
        return f"k{k}|{length_bucket}"

    @staticmethod
    def k_key(k: int) -> str:
        return f"k{k}"

    def score_step(
        self,
        magnitude: float,
        k: int,
        length_bucket: str,
        demarcator: str,
        n_min: int = DEFAULT_N_MIN,
    ) -> ConditionedScore:
        """Z-score + percentile a step's magnitude against its stratum.

        SPARSITY BACKOFF: try the finest ``(k, length, demarcator)`` cell first;
        if it is missing or has ``n < n_min``, back off to ``(k, length)``, then
        to ``(k)``. The level that answered is recorded on the result. Raises if
        even the ``(k)`` cell is missing or below ``n_min`` -- an honest failure
        rather than scoring against a cell too thin to calibrate.
        """
        for level, key in (
            ("k|length|demarcator", self.finest_key(k, length_bucket, demarcator)),
            ("k|length", self.length_key(k, length_bucket)),
            ("k", self.k_key(k)),
        ):
            stratum = self.strata.get(key)
            if stratum is not None and stratum.n >= n_min:
                return ConditionedScore(
                    z=stratum.zscore(magnitude),
                    percentile=stratum.percentile_rank(magnitude),
                    backoff_level=level,
                    stratum_key=key,
                    n=stratum.n,
                )
        raise ValueError(
            f"no stratum with n >= {n_min} for (k={k}, length={length_bucket!r}, "
            f"demarcator={demarcator!r}); even the k-level cell is too thin to "
            f"calibrate -- refusing to score against an under-populated null."
        )


def load_conditioned_null(path: str) -> ConditionedDisplacementNull:
    """Load a measured conditioned-phrase displacement null (ADR-SKM-0003).

    Hard-fails (no silent default) if the file is missing, the header lacks a
    required key, or the regime is not ``bearing-magnitude-conditioned``. No
    legacy-format reader (project rule).
    """
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Conditioned null artifact not found: {path}. "
            f"Build it first (no silent default)."
        ) from exc

    header = blob.get("header")
    if not isinstance(header, dict):
        raise ValueError(
            f"Conditioned null artifact {path} has no 'header' block; "
            f"refusing header-less null."
        )

    missing = [k for k in REQUIRED_CONDITIONED_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Conditioned null artifact {path} header missing required keys "
            f"{missing}; refusing to load an under-described null."
        )

    if header["regime"] != EXPECTED_CONDITIONED_REGIME:
        raise ValueError(
            f"Conditioned null artifact {path} regime is {header['regime']!r}, "
            f"expected {EXPECTED_CONDITIONED_REGIME!r}. "
            f"Wrong-regime null is a miscalibration."
        )

    raw_strata = blob.get("strata")
    if not isinstance(raw_strata, dict) or not raw_strata:
        raise ValueError(
            f"Conditioned null artifact {path} has no 'strata' block; "
            f"a stratified null without strata cannot score anything."
        )

    strata: Dict[str, ConditionedStratum] = {}
    for key, cell in raw_strata.items():
        mags = np.sort(np.asarray(cell.get("sorted_magnitudes", []), dtype=float))
        strata[key] = ConditionedStratum(
            mean=float(cell["mean"]),
            std=float(cell["std"]),
            n=int(cell["n"]),
            sorted_magnitudes=mags,
            percentiles=cell.get("percentiles", {}),
        )

    return ConditionedDisplacementNull(
        regime=header["regime"],
        atom=header["atom"],
        embedder=header["embedder"],
        dim=int(header["dim"]),
        k_range=list(header["k_range"]),
        strata=strata,
        header=header,
    )
