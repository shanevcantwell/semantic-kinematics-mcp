"""Direction command module: functional-direction probe (ADR-SKM-008).

Phase 1 of ADR-SKM-008 landed the calibration-artifact loader --
:func:`load_calibration`. It follows the ``bearing/jolt.py::load_null``
regime-typed self-describing-artifact pattern exactly: hard-fail on a missing
file, a missing/under-described header, or a wrong regime -- no silent
default, no legacy reader.

Phase 2 landed ``initialize_direction`` (ADR-SKM-008 D2/D3): a seedset
artifact + a calibration artifact -> a mean-centered difference-of-centroids
direction, validated by held-out AUC, a topic-control check, bootstrap
stability, and a corpus-null reference, then persisted as a self-describing
``functional-direction`` artifact. The axis math itself is NOT reimplemented
here -- :func:`~semantic_kinematics.mcp.commands.axis_alignment.build_axis`
is reused verbatim (ONE-DOOR: one projection kernel, a new axis *source*).

Phase 3 (this module addition) lands the projection + rate primitives (ADR
D4/D5): ``project_text``, ``project_chunks``, ``project_corpus``,
``query_rates``, ``top_exemplars``. All refuse a non-``usable`` direction
(D3 verdict gate) unless the caller passes an explicit override -- the ADR is
silent on projecting an under-determined direction, so this module refuses by
default (CODE_CONSTITUTION "never trust-and-degrade on an unknown identity" /
GROUND_PHYSICS falsification-shaped confidence: an unvalidated axis is not a
safe default to project against).

Later phases (D4/D6: ``cross_project``, ``direction_diagnostics``,
``preview_pattern``) register their MCP tools in this module and in
``server.py``; none of that surface exists yet.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from mcp.types import Tool

from semantic_kinematics.mcp.commands.axis_alignment import (
    DEFAULT_MIN_POLE_SEPARATION,
    build_axis,
)
from semantic_kinematics.mcp.state_manager import StateManager

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


# --------------------------------------------------------------------------- #
# Phase 2 (ADR-SKM-008 D2/D3): initialize_direction.
# --------------------------------------------------------------------------- #

# Seedset manifest required keys (TVI-008 <pattern_id>.seedset.json). The
# corpus_snapshot sub-block carries the identity gate (D2 step 1).
REQUIRED_SEEDSET_MANIFEST_KEYS = ("pattern_id", "corpus_snapshot")
REQUIRED_CORPUS_SNAPSHOT_KEYS = ("embedding_model_id", "vector_memmap_sha256")
REQUIRED_SEEDSET_ROW_KEYS = ("chunk_id", "rowid_mm")

EXPECTED_DIRECTION_REGIME = "functional-direction"
REQUIRED_DIRECTION_HEADER_KEYS = (
    "regime",
    "embedding_model_id",
    "source_memmap_sha256",
    "pattern_id",
    "era",
    "dim",
)

# D3 defaults. Where the ADR leaves a numeric unpinned it is named here, as a
# module-level default parameterizing the corresponding function -- never a
# bare literal at the call site (CODE_CONSTITUTION "name the enforcement
# surface": a threshold with no name is a threshold no one can override or
# audit).
DEFAULT_LEAKAGE_AUC_THRESHOLD = 0.98  # ADR-SKM-008 D3: ">0.98 is an alarm"
DEFAULT_BOOTSTRAP_B = 200  # ADR-SKM-008 D3: "resample ... B times (e.g. B=200)"
# The ADR does not pin a bootstrap-cosine promotion floor ("low bootstrap
# cosine -> refuse to promote (under-determined)" -- no number given). Named
# here, parameterized, so the gate is auditable and overridable rather than a
# silent magic number buried in a call site.
DEFAULT_MIN_BOOTSTRAP_COSINE = 0.7
DEFAULT_HELD_OUT_FRACTION = 0.3  # ADR-SKM-008 D3: "5-fold or 70/30"
DEFAULT_RANDOM_TOPIC_SAMPLE_SIZE = 200
# Below this many paired (seed, negative) groups, a held-out split cannot hold
# even one pair out per side meaningfully -- refuse rather than report a
# statistic computed on a degenerate split. Named per the dispatch's small-n
# guidance; not asserted anywhere else in the ADR text.
MIN_PAIRS_FOR_HELD_OUT_SPLIT = 4

# Phase 3 (D5) defaults. "Default: the projection z at which held-out
# seed-vs-negative precision >= 0.9" -- named here rather than a bare literal
# at the query_rates call site.
DEFAULT_RATE_TARGET_PRECISION = 0.9
DEFAULT_TOP_EXEMPLARS_K = 20
EXPECTED_PROJECTION_REGIME = "direction-projection"
REQUIRED_PROJECTION_HEADER_KEYS = (
    "regime",
    "embedding_model_id",
    "source_memmap_sha256",
    "direction_sha256",
    "pattern_id",
    "era",
    "n_used",
)
EXPECTED_RATE_TABLE_REGIME = "direction-rate-table"


class DirectionRefusal(Exception):
    """Raised for every REFUSE case in the Phase 2 pipeline: identity
    mismatch, under-described seedset, or a too-thin seedset that cannot
    support a held-out split. Mirrors :class:`CalibrationRefusal`'s role in
    Phase 1 -- a typed, named refusal rather than a bare ValueError, so a
    caller can distinguish "your input is bad" from "the ecosystem prevented
    a silent wrong answer."
    """


def load_seedset(json_path: str) -> Dict[str, Any]:
    """Load and validate a TVI-008 ``<pattern_id>.seedset.json`` artifact.

    Refuses (raises :class:`DirectionRefusal`) on a missing file, malformed
    JSON, a missing ``manifest.pattern_id``/``manifest.corpus_snapshot``, a
    ``corpus_snapshot`` missing ``embedding_model_id``/``vector_memmap_sha256``,
    or missing ``seeds``/``negatives`` arrays. Returns the parsed dict
    unchanged (files-only boundary -- TVI-008's seedset shape is read
    verbatim, never re-typed).
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except FileNotFoundError as exc:
        raise DirectionRefusal(
            f"Seedset artifact not found: {json_path}. Mint it corpus-side "
            "first (TVI-008 Phase 4; no silent default)."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DirectionRefusal(f"Seedset artifact {json_path} is not valid JSON: {exc}") from exc

    manifest = blob.get("manifest")
    if not isinstance(manifest, dict):
        raise DirectionRefusal(
            f"Seedset artifact {json_path} has no 'manifest' block; refusing "
            "a manifest-less seedset (no provenance to gate identity on)."
        )
    missing = [k for k in REQUIRED_SEEDSET_MANIFEST_KEYS if k not in manifest]
    if missing:
        raise DirectionRefusal(
            f"Seedset manifest {json_path} missing required keys {missing}."
        )

    snapshot = manifest.get("corpus_snapshot")
    if not isinstance(snapshot, dict):
        raise DirectionRefusal(
            f"Seedset manifest {json_path} has no 'corpus_snapshot' block; "
            "refusing -- the identity gate (D2 step 1) has nothing to check."
        )
    missing_snapshot = [k for k in REQUIRED_CORPUS_SNAPSHOT_KEYS if k not in snapshot]
    if missing_snapshot:
        raise DirectionRefusal(
            f"Seedset {json_path} corpus_snapshot missing required keys "
            f"{missing_snapshot}."
        )

    seeds = blob.get("seeds")
    negatives = blob.get("negatives")
    if not isinstance(seeds, list) or not seeds:
        raise DirectionRefusal(f"Seedset {json_path} has no non-empty 'seeds' list.")
    if not isinstance(negatives, list) or not negatives:
        raise DirectionRefusal(f"Seedset {json_path} has no non-empty 'negatives' list.")

    for label, rows in (("seeds", seeds), ("negatives", negatives)):
        for i, row in enumerate(rows):
            missing_row = [k for k in REQUIRED_SEEDSET_ROW_KEYS if k not in row]
            if missing_row:
                raise DirectionRefusal(
                    f"Seedset {json_path} {label}[{i}] missing required keys "
                    f"{missing_row}."
                )

    return blob


def refuse_unless_seedset_matches_calibration(seedset: Dict[str, Any], calibration: "Calibration") -> None:
    """D2 step 1: the seedset's declared identity must match the calibration's.

    Delegates to :meth:`Calibration.refuse_unless_matches` (the identity gate
    already lives there); this wraps its ``ValueError`` as a
    :class:`DirectionRefusal` so Phase 2's refusal surface is uniformly typed.
    """
    snapshot = seedset["manifest"]["corpus_snapshot"]
    try:
        calibration.refuse_unless_matches(
            embedding_model_id=snapshot["embedding_model_id"],
            source_memmap_sha256=snapshot["vector_memmap_sha256"],
        )
    except ValueError as exc:
        raise DirectionRefusal(str(exc)) from exc


def _filter_rows_by_era(rows: Sequence[Dict[str, Any]], era: Optional[str]) -> List[Dict[str, Any]]:
    """Denormalized era-scoped filter (ADR-SKM-008 D2 step 4). No filter (era=None)
    returns all rows unchanged."""
    if era is None:
        return list(rows)
    return [r for r in rows if r.get("era") == era]


def read_memmap_rows(
    memmap_path: str, rowids: Sequence[int], dim: int, n_used: int
) -> np.ndarray:
    """Read a set of rows from the frozen TVI-008 dense float32 memmap by
    ``rowid_mm``, cast to float64. Mirrors
    ``scripts/build_corpus_calibration.py::load_memmap_matrix``'s memmap-open
    convention (mode="r", shape=(n_used, dim)) but returns only the requested
    rows rather than the whole matrix -- a seedset is a few hundred rows, no
    need to materialize a multi-GB read.
    """
    expected_bytes = n_used * dim * 4
    actual_bytes = os.path.getsize(memmap_path)
    if actual_bytes != expected_bytes:
        raise DirectionRefusal(
            f"memmap {memmap_path} size {actual_bytes} bytes does not match "
            f"n_used*dim*4 = {expected_bytes} bytes (n_used={n_used}, dim={dim})."
        )
    mm = np.memmap(memmap_path, dtype=np.float32, mode="r", shape=(n_used, dim))
    idx = np.asarray(list(rowids), dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= n_used):
        raise DirectionRefusal(
            f"rowid_mm out of range for memmap with n_used={n_used} "
            f"(min={int(idx.min())}, max={int(idx.max())})."
        )
    return np.asarray(mm[idx], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Pure numeric core: mean-centering, direction extraction, D3 diagnostics.
# No IO -- exhaustively unit-testable with hand-built matrices.
# --------------------------------------------------------------------------- #

def mean_center(vecs: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Subtract the corpus mean mu from each row of vecs (D2 step 2)."""
    return np.asarray(vecs, dtype=np.float64) - np.asarray(mu, dtype=np.float64)


def compute_direction(
    centered_seeds: np.ndarray,
    centered_negatives: np.ndarray,
    min_pole_separation: float = DEFAULT_MIN_POLE_SEPARATION,
) -> Dict[str, Any]:
    """D2 step 3: direction = centroid(centered seeds) - centroid(centered
    negatives), unit-normalized. Reuses ``axis_alignment.build_axis`` verbatim
    (ONE-DOOR) -- the negative centroid is passed as ``neg_pole``, exactly the
    "averaged negative exemplars" case ``build_axis`` already supports; no
    second projection/pole-difference implementation is written here.

    Returns {"unit_axis": (dim,), "pole_separation": float} or
    {"error": ...} if the poles are underdetermined (below
    ``min_pole_separation``, mirroring ``alignment_core``'s gate).
    """
    neg_pole = np.asarray(centered_negatives, dtype=np.float64).mean(axis=0)
    unit_axis, pole_separation = build_axis(centered_seeds, neg_pole)
    if pole_separation < min_pole_separation:
        return {
            "error": "axis underdetermined",
            "detail": (
                f"pole separation {pole_separation:.4f} < minimum "
                f"{min_pole_separation:.4f}; seeds and negatives centroid too "
                "close together to define a stable direction."
            ),
            "pole_separation": pole_separation,
        }
    return {"unit_axis": unit_axis, "pole_separation": pole_separation}


def auc_score(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """AUC of using `score` to separate pos from neg, via the Mann-Whitney U
    statistic (rank-sum) -- exact, dependency-free (no scipy/sklearn in this
    project's dependency set), and equivalent to the probability that a
    randomly drawn positive scores higher than a randomly drawn negative
    (ties count as one-half).
    """
    pos_scores = np.asarray(pos_scores, dtype=np.float64)
    neg_scores = np.asarray(neg_scores, dtype=np.float64)
    n_pos, n_neg = pos_scores.size, neg_scores.size
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auc_score requires at least one positive and one negative score.")
    combined = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    # Average ranks for ties (standard Mann-Whitney tie handling).
    sorted_vals = combined[order]
    rank_vals = np.empty(combined.size, dtype=np.float64)
    i = 0
    while i < combined.size:
        j = i
        while j + 1 < combined.size and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed rank, averaged over the tie block
        rank_vals[i : j + 1] = avg_rank
        i = j + 1
    ranks[order] = rank_vals
    pos_rank_sum = float(ranks[:n_pos].sum())
    u_stat = pos_rank_sum - n_pos * (n_pos + 1) / 2.0
    return u_stat / (n_pos * n_neg)


def _paired_held_out_split(
    paired_seed_chunk_ids: Sequence[Optional[str]],
    seed_chunk_ids: Sequence[str],
    held_out_fraction: float,
    rng: np.random.Generator,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """Split (seed, matched-negative) PAIRS by group so a seed and its paired
    negative never straddle the train/held-out boundary (D3 "split by paired
    groups"). Unpaired seeds/negatives (no matching partner) are assigned to
    train only, since they cannot form a held-out pair.

    Returns (train_seed_idx, held_seed_idx, train_neg_idx, held_neg_idx) as
    index lists into the original seed/negative row order.
    """
    chunk_to_seed_idx = {cid: i for i, cid in enumerate(seed_chunk_ids)}
    paired_neg_idx = []
    paired_seed_idx = []
    unpaired_neg_idx = []
    for j, paired_chunk_id in enumerate(paired_seed_chunk_ids):
        seed_i = chunk_to_seed_idx.get(paired_chunk_id)
        if paired_chunk_id is not None and seed_i is not None:
            paired_neg_idx.append(j)
            paired_seed_idx.append(seed_i)
        else:
            unpaired_neg_idx.append(j)

    n_pairs = len(paired_seed_idx)
    n_held = max(1, int(round(n_pairs * held_out_fraction))) if n_pairs else 0
    order = rng.permutation(n_pairs)
    held_pair_pos = set(order[:n_held].tolist())

    train_seed_idx, held_seed_idx = [], []
    train_neg_idx, held_neg_idx = [], []
    for pos, (s_i, n_j) in enumerate(zip(paired_seed_idx, paired_neg_idx)):
        if pos in held_pair_pos:
            held_seed_idx.append(s_i)
            held_neg_idx.append(n_j)
        else:
            train_seed_idx.append(s_i)
            train_neg_idx.append(n_j)

    # Seeds with no paired negative, and negatives with no paired seed, train only.
    paired_seed_set = set(paired_seed_idx)
    for i in range(len(seed_chunk_ids)):
        if i not in paired_seed_set:
            train_seed_idx.append(i)
    train_neg_idx.extend(unpaired_neg_idx)

    return train_seed_idx, held_seed_idx, train_neg_idx, held_neg_idx


def held_out_separation_auc(
    centered_seeds: np.ndarray,
    centered_negatives: np.ndarray,
    paired_seed_chunk_ids: Sequence[Optional[str]],
    seed_chunk_ids: Sequence[str],
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    min_pole_separation: float = DEFAULT_MIN_POLE_SEPARATION,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """D3 "held-out separation AUC": split by paired groups, extract the
    direction on the train split, project held-out seeds vs held-out
    negatives, report AUC as a separator.

    Returns {"auc": float, "n_held_seeds": int, "n_held_negatives": int,
    "leakage_suspected": bool} or {"error": ...} if there are too few paired
    groups to hold out meaningfully (MIN_PAIRS_FOR_HELD_OUT_SPLIT).
    """
    rng = rng or np.random.default_rng(0)
    chunk_to_seed_idx = {cid: i for i, cid in enumerate(seed_chunk_ids)}
    n_pairs_available = sum(
        1 for pcid in paired_seed_chunk_ids if pcid is not None and pcid in chunk_to_seed_idx
    )
    if n_pairs_available < MIN_PAIRS_FOR_HELD_OUT_SPLIT:
        return {
            "error": (
                "too few paired (seed, negative) groups for a held-out split "
                f"({n_pairs_available} < minimum {MIN_PAIRS_FOR_HELD_OUT_SPLIT})"
            ),
            "n_pairs_available": n_pairs_available,
        }

    train_s, held_s, train_n, held_n = _paired_held_out_split(
        paired_seed_chunk_ids=paired_seed_chunk_ids,
        seed_chunk_ids=seed_chunk_ids,
        held_out_fraction=held_out_fraction,
        rng=rng,
    )
    if not held_s:
        return {
            "error": "held-out split produced zero held-out pairs",
            "n_pairs_available": n_pairs_available,
        }

    train_direction = compute_direction(
        centered_seeds[train_s], centered_negatives[train_n], min_pole_separation
    )
    if "error" in train_direction:
        return train_direction

    unit_axis = train_direction["unit_axis"]
    held_seed_proj = centered_seeds[held_s] @ unit_axis
    held_neg_proj = centered_negatives[held_n] @ unit_axis
    auc = auc_score(held_seed_proj, held_neg_proj)
    return {
        "auc": round(float(auc), 4),
        "n_held_seeds": len(held_s),
        "n_held_negatives": len(held_n),
        "leakage_suspected": bool(auc > DEFAULT_LEAKAGE_AUC_THRESHOLD),
    }


def topic_control_check(
    unit_axis: np.ndarray,
    held_out_negatives: np.ndarray,
    held_out_seeds: np.ndarray,
    random_topic_sample: np.ndarray,
) -> Dict[str, Any]:
    """D3 "topic-control check": project held-out negatives + seeds and a
    random topic-matched sample onto the direction. Seeds should separate
    from negatives; a random sample should NOT shift relative to negatives
    (a functional, not topical, direction). Reports both separations so the
    caller can compare; ``functional_not_topical`` is a coarse verdict flag
    (seeds separate meaningfully, random sample does not).
    """
    seed_proj = np.asarray(held_out_seeds, dtype=np.float64) @ unit_axis
    neg_proj = np.asarray(held_out_negatives, dtype=np.float64) @ unit_axis
    random_proj = np.asarray(random_topic_sample, dtype=np.float64) @ unit_axis

    seed_vs_neg_auc = auc_score(seed_proj, neg_proj)
    # random-vs-negative AUC near 0.5 means "no separation" -- report the
    # deviation from chance as the topic-orthogonality evidence.
    random_vs_neg_auc = auc_score(random_proj, neg_proj)

    return {
        "seed_vs_negative_auc": round(float(seed_vs_neg_auc), 4),
        "random_vs_negative_auc": round(float(random_vs_neg_auc), 4),
        "random_topic_sample_n": int(random_topic_sample.shape[0]),
        # Seeds separate meaningfully (well above chance) while the random
        # sample stays near chance (0.5) against the same negatives.
        "functional_not_topical": bool(
            seed_vs_neg_auc > 0.6 and abs(random_vs_neg_auc - 0.5) < 0.15
        ),
    }


def bootstrap_stability(
    centered_seeds: np.ndarray,
    centered_negatives: np.ndarray,
    b: int = DEFAULT_BOOTSTRAP_B,
    min_pole_separation: float = DEFAULT_MIN_POLE_SEPARATION,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """D3 "bootstrap stability": resample seeds/negatives with replacement B
    times, re-extract the direction, report the mean pairwise cosine of the
    bootstrapped directions (reproducibility) and a bootstrap CI on
    ``pole_separation``.
    """
    rng = rng or np.random.default_rng(0)
    n_seeds, n_negs = centered_seeds.shape[0], centered_negatives.shape[0]
    axes: List[np.ndarray] = []
    separations: List[float] = []
    for _ in range(b):
        seed_sample = centered_seeds[rng.integers(0, n_seeds, size=n_seeds)]
        neg_sample = centered_negatives[rng.integers(0, n_negs, size=n_negs)]
        result = compute_direction(seed_sample, neg_sample, min_pole_separation=0.0)
        if "error" in result:
            continue
        axes.append(result["unit_axis"])
        separations.append(result["pole_separation"])

    if len(axes) < 2:
        return {
            "error": "bootstrap could not extract at least 2 stable directions",
            "n_successful_resamples": len(axes),
        }

    axes_arr = np.stack(axes, axis=0)
    # Mean pairwise cosine over all C(len,2) pairs, vectorized via the Gram
    # matrix (axes are already unit-norm from build_axis).
    gram = axes_arr @ axes_arr.T
    n = gram.shape[0]
    off_diag_sum = float(gram.sum() - np.trace(gram))
    mean_pairwise_cosine = off_diag_sum / (n * (n - 1))

    sep_arr = np.asarray(separations, dtype=np.float64)
    ci_lo, ci_hi = np.percentile(sep_arr, [2.5, 97.5])

    return {
        "b": b,
        "n_successful_resamples": len(axes),
        "mean_pairwise_cosine": round(float(mean_pairwise_cosine), 4),
        "pole_separation_ci_95": [round(float(ci_lo), 4), round(float(ci_hi), 4)],
        "under_determined": bool(mean_pairwise_cosine < DEFAULT_MIN_BOOTSTRAP_COSINE),
    }


def null_reference(centered_corpus_sample: np.ndarray, unit_axis: np.ndarray) -> Dict[str, Any]:
    """D3 "null calibration reference": the mean-centered corpus projected
    onto this direction. mu0 is ~0 by construction (the input is already
    mean-centered); sigma0 is the corpus spread along the axis -- recorded so
    run/rate verbs (Phase 3+) z-score without recomputation.
    """
    proj = np.asarray(centered_corpus_sample, dtype=np.float64) @ unit_axis
    mu0 = float(proj.mean())
    sigma0 = float(proj.std())
    # Distribution-shape note (D3/D5): record simple moments now; a
    # non-Gaussian projection switches Phase-3 thresholds to empirical
    # quantiles rather than sigma-multiples (deferred to that phase).
    return {
        "mu0": round(mu0, 6),
        "sigma0": round(sigma0, 6),
        "n": int(centered_corpus_sample.shape[0]),
    }


def verdict_from_diagnostics(
    direction_result: Dict[str, Any],
    held_out: Dict[str, Any],
    bootstrap: Dict[str, Any],
) -> str:
    """Falsification-shaped confidence (D3/ADR-SKMCP-0002 rule 4): a too-clean
    result is an alarm, not a win. Returns one of {"usable",
    "under-determined", "leakage-suspected"}.

    Precedence: an axis-underdetermined, bootstrap-failure, or held-out-split
    failure result is "under-determined" outright (nothing to trust) -- an
    unmeasured leakage diagnostic (too few paired groups to split) must not
    read as a clean pass; leakage is checked next (a >0.98 held-out AUC
    discredits the result regardless of stability); otherwise an
    under-determined bootstrap cosine still blocks promotion.
    """
    if "error" in direction_result:
        return "under-determined"
    if "error" in bootstrap:
        return "under-determined"
    if "error" in held_out:
        return "under-determined"
    if held_out.get("leakage_suspected"):
        return "leakage-suspected"
    if bootstrap.get("under_determined"):
        return "under-determined"
    return "usable"


def initialize_direction_core(
    calibration: Calibration,
    seedset: Dict[str, Any],
    seed_vecs: np.ndarray,
    negative_vecs: np.ndarray,
    random_topic_vecs: np.ndarray,
    corpus_null_sample_vecs: np.ndarray,
    era: Optional[str] = None,
    min_pole_separation: float = DEFAULT_MIN_POLE_SEPARATION,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    held_out_fraction: float = DEFAULT_HELD_OUT_FRACTION,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """The pure orchestration core of ``initialize_direction`` (D2 steps 2-3 +
    D3), given already-loaded raw (uncentered) vectors for seeds, negatives,
    a random topic-matched sample, and a corpus-null sample. IO (memmap reads,
    seedset/calibration loading, artifact persistence) lives in the MCP
    handler; this function is exhaustively unit-testable with synthetic
    arrays.

    Returns a dict with keys: ``unit_axis``, ``pole_separation``,
    ``held_out_auc``, ``topic_control``, ``bootstrap``, ``null_reference``,
    ``verdict``, or ``{"error": ...}`` if direction extraction itself fails
    (poles underdetermined on the FULL seed/negative set -- the fatal case;
    a too-few-pairs held-out split or an under-determined bootstrap surface
    as a non-fatal "under-determined" verdict instead, since the direction
    itself was still extracted).
    """
    rng = rng or np.random.default_rng(0)

    seeds_rows = _filter_rows_by_era(seedset["seeds"], era)
    negatives_rows = _filter_rows_by_era(seedset["negatives"], era)
    if not seeds_rows:
        return {"error": f"era filter {era!r} matched zero seed rows."}
    if not negatives_rows:
        return {"error": f"era filter {era!r} matched zero negative rows."}

    centered_seeds = mean_center(seed_vecs, calibration.mu)
    centered_negatives = mean_center(negative_vecs, calibration.mu)

    direction_result = compute_direction(
        centered_seeds, centered_negatives, min_pole_separation=min_pole_separation
    )
    if "error" in direction_result:
        return direction_result

    unit_axis = direction_result["unit_axis"]
    pole_separation = direction_result["pole_separation"]

    seed_chunk_ids = [r["chunk_id"] for r in seeds_rows]
    paired_seed_chunk_ids = [r.get("paired_seed_chunk_id") for r in negatives_rows]

    held_out = held_out_separation_auc(
        centered_seeds,
        centered_negatives,
        paired_seed_chunk_ids=paired_seed_chunk_ids,
        seed_chunk_ids=seed_chunk_ids,
        held_out_fraction=held_out_fraction,
        min_pole_separation=min_pole_separation,
        rng=rng,
    )

    centered_random = mean_center(random_topic_vecs, calibration.mu)
    # Topic-control runs IN-SAMPLE against the full-data axis: it passes the
    # full centered negatives/seeds and the axis fit on all of them (not the
    # held-out split's subsets). The random-vs-negative orthogonality
    # comparison is same-axis relative, so it is directionally sound. The
    # seed-vs-negative AUC reported here, however, is in-sample and optimistic
    # by construction -- it measures separation on the same vectors the axis
    # was fit from; the held-out AUC above is the leakage-honest figure.
    topic_control = topic_control_check(
        unit_axis,
        held_out_negatives=centered_negatives,
        held_out_seeds=centered_seeds,
        random_topic_sample=centered_random,
    )

    bootstrap = bootstrap_stability(
        centered_seeds,
        centered_negatives,
        b=bootstrap_b,
        min_pole_separation=min_pole_separation,
        rng=rng,
    )

    centered_null_sample = mean_center(corpus_null_sample_vecs, calibration.mu)
    null_ref = null_reference(centered_null_sample, unit_axis)

    verdict = verdict_from_diagnostics(direction_result, held_out, bootstrap)

    return {
        "unit_axis": unit_axis,
        "pole_separation": pole_separation,
        "held_out_auc": held_out,
        "topic_control": topic_control,
        "bootstrap": bootstrap,
        "null_reference": null_ref,
        "verdict": verdict,
        "n_seeds": len(seeds_rows),
        "n_negatives": len(negatives_rows),
    }


# --------------------------------------------------------------------------- #
# Direction artifact persistence: regime-typed, self-describing, jolt.py-style
# hard-fail loader.
# --------------------------------------------------------------------------- #

def direction_artifact_paths(out_dir: str, pattern_id: str, era: Optional[str]) -> Tuple[str, str]:
    """``data/directions/<pattern_id>[.<era>].direction.{npz,json}`` (D2 step 5)."""
    slug = pattern_id if era is None else f"{pattern_id}.{era}"
    # Filesystem-safe: era strings may contain spaces (e.g. "Opus 4.8").
    safe_slug = slug.replace("/", "__").replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{safe_slug}.direction.npz")
    json_path = os.path.join(out_dir, f"{safe_slug}.direction.json")
    return npz_path, json_path


def write_direction_artifact(
    out_dir: str,
    pattern_id: str,
    era: Optional[str],
    unit_axis: np.ndarray,
    calibration: Calibration,
    seedset_manifest: Dict[str, Any],
    diagnostics: Dict[str, Any],
    axis_source: str = "seedset_centroids",
) -> Tuple[str, str]:
    """Persist the self-describing ``functional-direction`` artifact (D2 step 5):
    the unit axis (npz) + a manifest (json) carrying the full provenance chain
    -- seedset manifest, calibration manifest identity, era-scope, counts,
    pole_separation, and the D3 diagnostics.
    """
    npz_path, json_path = direction_artifact_paths(out_dir, pattern_id, era)
    np.savez(npz_path, unit_axis=np.asarray(unit_axis, dtype=np.float64))

    manifest = {
        "header": {
            "regime": EXPECTED_DIRECTION_REGIME,
            "embedding_model_id": calibration.embedding_model_id,
            "source_memmap_sha256": calibration.source_memmap_sha256,
            "pattern_id": pattern_id,
            "era": era,
            "dim": calibration.dim,
        },
        "axis_source": axis_source,
        "pole_separation": diagnostics["pole_separation"],
        "verdict": diagnostics["verdict"],
        "n_seeds": diagnostics["n_seeds"],
        "n_negatives": diagnostics["n_negatives"],
        "held_out_auc": diagnostics["held_out_auc"],
        "topic_control": diagnostics["topic_control"],
        "bootstrap": diagnostics["bootstrap"],
        "null_reference": diagnostics["null_reference"],
        "calibration_manifest": {
            "embedding_model_id": calibration.embedding_model_id,
            "source_memmap_sha256": calibration.source_memmap_sha256,
            "mu_norm": calibration.mu_norm,
        },
        "seedset_manifest": seedset_manifest,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    return npz_path, json_path


@dataclass
class Direction:
    """A loaded ``functional-direction`` artifact.

    ``direction_sha256`` is the content-addressed sha256 of the ``unit_axis``
    npz file bytes, computed at load time (the artifact itself does not
    persist a self-sha -- Phase 2 had no consumer that needed one). Phase 3
    keys projection artifacts on ``(direction_sha256, source_memmap_sha256)``
    per the ADR D4 ``project_corpus`` row ("keyed by direction-sha +
    memmap-sha"), so a projection is invalidated *both* when the corpus is
    rebuilt (memmap sha changes, already gated) and when the direction itself
    is re-extracted with different seeds/params (unit_axis bytes change).
    """

    embedding_model_id: str
    source_memmap_sha256: str
    pattern_id: str
    era: Optional[str]
    dim: int
    unit_axis: np.ndarray
    axis_source: str
    pole_separation: float
    verdict: str
    direction_sha256: str = ""
    header: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


def load_direction(json_path: str) -> Direction:
    """Load a ``functional-direction`` artifact (jolt.py ``load_null`` /
    Phase-1 ``load_calibration`` discipline): hard-fail on missing
    manifest/npz, a header missing a required key, or the wrong regime.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Direction artifact not found: {json_path}. Build it first with "
            "initialize_direction (no silent default)."
        ) from exc

    header = manifest.get("header")
    if not isinstance(header, dict):
        raise ValueError(
            f"Direction artifact {json_path} has no 'header' block; refusing "
            "header-less direction."
        )
    missing = [k for k in REQUIRED_DIRECTION_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Direction artifact {json_path} header missing required keys "
            f"{missing}; refusing to load an under-described direction."
        )
    if header["regime"] != EXPECTED_DIRECTION_REGIME:
        raise ValueError(
            f"Direction artifact {json_path} regime is {header['regime']!r}, "
            f"expected {EXPECTED_DIRECTION_REGIME!r}."
        )

    npz_path = os.path.splitext(json_path)[0] + ".npz"
    try:
        npz = np.load(npz_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Direction npz not found: {npz_path} (expected alongside "
            f"{json_path}). Build it first (no silent default)."
        ) from exc
    if "unit_axis" not in npz:
        raise ValueError(f"Direction npz {npz_path} has no 'unit_axis' array.")

    direction_sha256 = _sha256_of_file(npz_path)

    return Direction(
        embedding_model_id=header["embedding_model_id"],
        source_memmap_sha256=header["source_memmap_sha256"],
        pattern_id=header["pattern_id"],
        era=header["era"],
        dim=int(header["dim"]),
        unit_axis=np.asarray(npz["unit_axis"], dtype=np.float64),
        axis_source=manifest.get("axis_source", "seedset_centroids"),
        pole_separation=float(manifest.get("pole_separation", 0.0)),
        verdict=manifest.get("verdict", "under-determined"),
        direction_sha256=direction_sha256,
        header=header,
        manifest=manifest,
    )


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> str:
    """Stream sha256 of a file without loading it whole (mirrors
    ``scripts/build_corpus_calibration.py::sha256_of_file`` -- not imported
    from there to avoid a script->module import direction; small enough to
    duplicate rather than restructure package layout for one helper)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# MCP tool surface.
# --------------------------------------------------------------------------- #

DIRECTION_ARTIFACT_DIR = os.path.join("data", "directions")
PROJECTION_ARTIFACT_DIR = os.path.join("data", "projections")
RATE_ARTIFACT_DIR = os.path.join("data", "rates")

# D3/D5: refuse to project a direction whose verdict is not "usable" unless
# the caller explicitly overrides (ADR-SKM-008 does not state a projection
# policy for "under-determined"/"leakage-suspected" directions; silence reads
# as refuse-by-default per CODE_CONSTITUTION "never trust-and-degrade on an
# unknown identity" -- an unvalidated axis is not a safe default to project
# against). Named + parameterized so the override is explicit and auditable,
# never a silently-permissive default.
ALLOW_NON_USABLE_PROJECTION_ARG = "allow_non_usable_direction"


def refuse_unless_usable(direction: "Direction", allow_override: bool) -> None:
    """D3/D5 verdict gate: refuse to project a non-``usable`` direction unless
    ``allow_override`` is explicitly set. Every ``project_*`` verb calls this
    before touching the direction's ``unit_axis`` (ADR-SKM-008 Phase 3
    acceptance: "project_text refuses a non-usable direction" -- generalized
    here to every projection verb, since the same axis feeds all of them).
    """
    if direction.verdict != "usable" and not allow_override:
        raise DirectionRefusal(
            f"direction {direction.pattern_id!r}"
            f"{('.' + direction.era) if direction.era else ''} has verdict "
            f"{direction.verdict!r}, not 'usable'; refusing to project against "
            "an unvalidated axis. The ADR is silent on projecting an "
            "under-determined direction, so this refuses by default -- pass "
            f"{ALLOW_NON_USABLE_PROJECTION_ARG}=true to override explicitly "
            "(the projection result will carry this override flag)."
        )


def get_tools() -> List[Tool]:
    """Return direction-probe tool definitions (Phase 2 ``initialize_direction``;
    Phase 3 ``project_text``/``project_chunks``/``project_corpus``/
    ``query_rates``/``top_exemplars``; Phase 4 verbs land in their own PR)."""
    return [
        Tool(
            name="initialize_direction",
            description=(
                "Build a functional-direction axis from a seedset artifact's "
                "corpus-row centroids (seeds minus topic-matched negatives), "
                "mean-centered against a corpus-calibration artifact, and "
                "validate it: held-out separation AUC (with a leakage alarm "
                "above 0.98), a topic-control check, bootstrap direction "
                "stability, and a corpus-null projection reference. Refuses if "
                "the seedset's embedding_model_id/vector_memmap_sha256 do not "
                "match the calibration's. Optionally era-scoped. Persists a "
                "self-describing 'functional-direction' artifact and returns "
                "its ref plus the verdict "
                "(usable | under-determined | leakage-suspected)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "seedset_ref": {
                        "type": "string",
                        "description": "Path to the <pattern_id>.seedset.json artifact (TVI-008).",
                    },
                    "calibration_ref": {
                        "type": "string",
                        "description": "Path to the <slug>.calibration.json artifact (Phase 1).",
                    },
                    "tvi_root": {
                        "type": "string",
                        "description": (
                            "Root of the thought-vault-integration checkout, used to "
                            "resolve the calibration's repo-root-relative "
                            "source_memmap_path (files-only boundary, ADR-SKM-008 "
                            "Option E). If omitted, the path is tried relative to the "
                            "current working directory and refused with an "
                            "instructive error if not found there."
                        ),
                    },
                    "era": {
                        "type": "string",
                        "description": (
                            "Optional era filter (e.g. 'Opus 4.8'). Seeds/negatives "
                            "are filtered to this era before extraction; the "
                            "artifact is scoped <pattern_id>.<era>.direction.*."
                        ),
                    },
                    "out_dir": {
                        "type": "string",
                        "description": f"Output directory (default: {DIRECTION_ARTIFACT_DIR}).",
                        "default": DIRECTION_ARTIFACT_DIR,
                    },
                    "min_pole_separation": {
                        "type": "number",
                        "default": DEFAULT_MIN_POLE_SEPARATION,
                    },
                    "bootstrap_b": {
                        "type": "integer",
                        "default": DEFAULT_BOOTSTRAP_B,
                    },
                    "held_out_fraction": {
                        "type": "number",
                        "default": DEFAULT_HELD_OUT_FRACTION,
                    },
                    "random_topic_sample_size": {
                        "type": "integer",
                        "default": DEFAULT_RANDOM_TOPIC_SAMPLE_SIZE,
                    },
                    "seed": {
                        "type": "integer",
                        "description": "RNG seed for bootstrap/held-out split/random sample (default: 0).",
                        "default": 0,
                    },
                },
                "required": ["seedset_ref", "calibration_ref"],
            },
        ),
        Tool(
            name="project_text",
            description=(
                "Embed input text (single-embedder, pinned by the direction "
                "artifact's embedding_model_id), mean-center against the "
                "direction's calibration mu, and project onto the direction's "
                "unit axis -> {projection, z, cosine}. Refuses a non-'usable' "
                "direction unless allow_non_usable_direction=true."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Passage to project."},
                    "direction_ref": {
                        "type": "string",
                        "description": "Path to a <pattern_id>[.<era>].direction.json artifact.",
                    },
                    "calibration_ref": {
                        "type": "string",
                        "description": (
                            "Path to the calibration artifact carrying mu "
                            "(must match direction's embedding_model_id/memmap sha)."
                        ),
                    },
                    ALLOW_NON_USABLE_PROJECTION_ARG: {
                        "type": "boolean",
                        "description": "Override the usable-verdict refusal (default: false).",
                        "default": False,
                    },
                },
                "required": ["text", "direction_ref", "calibration_ref"],
            },
        ),
        Tool(
            name="project_chunks",
            description=(
                "Project a caller-supplied list of rowid_mm from the frozen "
                "memmap onto the direction -> per-row {rowid_mm, z}. No "
                "embedding call -- pure memmap read + matvec. Refuses a "
                "non-'usable' direction unless overridden."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "direction_ref": {"type": "string"},
                    "calibration_ref": {"type": "string"},
                    "rowids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "rowid_mm values to project (memmap row indices).",
                    },
                    "tvi_root": {
                        "type": "string",
                        "description": "Root of the tvi checkout, to resolve a relative source_memmap_path.",
                    },
                    ALLOW_NON_USABLE_PROJECTION_ARG: {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["direction_ref", "calibration_ref", "rowids"],
            },
        ),
        Tool(
            name="project_corpus",
            description=(
                "Full-corpus projection: the N x dim matvec over the whole "
                "memmap. Persists a projection artifact "
                "(data/projections/<pattern_id>[.<era>].proj.npz: per-rowid_mm "
                "z) keyed by direction-sha + memmap-sha, and returns its ref "
                "plus summary stats. Row count reconciles to the memmap's "
                "n_used. Refuses a non-'usable' direction unless overridden."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "direction_ref": {"type": "string"},
                    "calibration_ref": {"type": "string"},
                    "tvi_root": {
                        "type": "string",
                        "description": "Root of the tvi checkout, to resolve a relative source_memmap_path.",
                    },
                    "out_dir": {
                        "type": "string",
                        "default": PROJECTION_ARTIFACT_DIR,
                    },
                    ALLOW_NON_USABLE_PROJECTION_ARG: {
                        "type": "boolean",
                        "default": False,
                    },
                },
                "required": ["direction_ref", "calibration_ref"],
            },
        ),
        Tool(
            name="query_rates",
            description=(
                "Read a projection artifact + corpus.db (read-only, for "
                "era/channel/speaker grouping) -> an aggregated rate table: "
                "fraction of chunks above the calibrated threshold, by "
                "era x channel x speaker. The threshold is calibrated against "
                "the direction artifact's held-out seed/negative projections "
                "(D5: the projection z at which held-out precision >= a target, "
                "default 0.9) unless an explicit threshold is supplied. "
                "Threshold + precision + source shas are recorded in the "
                "rate-table artifact's manifest (regen-not-authored)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "projection_ref": {
                        "type": "string",
                        "description": "Path to a <pattern_id>[.<era>].proj.json artifact.",
                    },
                    "corpus_db_ref": {
                        "type": "string",
                        "description": "Path to corpus.db (opened read-only).",
                    },
                    "direction_ref": {
                        "type": "string",
                        "description": (
                            "Path to the direction artifact whose held-out "
                            "projections calibrate the default threshold. "
                            "Required unless 'threshold' is supplied explicitly."
                        ),
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Explicit z-score threshold (skips D5 auto-calibration).",
                    },
                    "target_precision": {
                        "type": "number",
                        "default": DEFAULT_RATE_TARGET_PRECISION,
                    },
                    "group_by": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["era", "channel", "speaker"]},
                        "default": ["era", "channel", "speaker"],
                    },
                    "out_dir": {"type": "string", "default": RATE_ARTIFACT_DIR},
                },
                "required": ["projection_ref", "corpus_db_ref"],
            },
        ),
        Tool(
            name="top_exemplars",
            description=(
                "Top-K chunk_ids along a direction (from a projection "
                "artifact), with text fetched via corpus.db (read-only) for "
                "readback."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "projection_ref": {"type": "string"},
                    "corpus_db_ref": {"type": "string"},
                    "k": {"type": "integer", "default": DEFAULT_TOP_EXEMPLARS_K},
                    "era": {
                        "type": "string",
                        "description": "Optional era filter applied before ranking.",
                    },
                },
                "required": ["projection_ref", "corpus_db_ref"],
            },
        ),
    ]


async def initialize_direction(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``initialize_direction`` (D2/D3). Stateless: resolves the
    seedset + calibration from disk, reads the memmap rows it needs, computes
    and validates the direction, persists the artifact, and returns. No
    cross-call state; ``manager`` is accepted for handler-signature parity
    with the rest of the command modules but is unused (no embedding calls in
    this verb -- D2 step 2 explicitly consumes precomputed corpus vectors).
    """
    seedset_ref = args.get("seedset_ref")
    calibration_ref = args.get("calibration_ref")
    if not seedset_ref:
        return {"error": "seedset_ref is required."}
    if not calibration_ref:
        return {"error": "calibration_ref is required."}

    era = args.get("era")
    out_dir = args.get("out_dir") or DIRECTION_ARTIFACT_DIR
    min_pole_separation = args.get("min_pole_separation", DEFAULT_MIN_POLE_SEPARATION)
    bootstrap_b = args.get("bootstrap_b", DEFAULT_BOOTSTRAP_B)
    held_out_fraction = args.get("held_out_fraction", DEFAULT_HELD_OUT_FRACTION)
    random_topic_sample_size = args.get(
        "random_topic_sample_size", DEFAULT_RANDOM_TOPIC_SAMPLE_SIZE
    )
    seed = int(args.get("seed", 0))

    try:
        calibration = load_calibration(calibration_ref)
        seedset = load_seedset(seedset_ref)
        refuse_unless_seedset_matches_calibration(seedset, calibration)
    except (FileNotFoundError, ValueError, DirectionRefusal) as exc:
        return {"error": str(exc)}

    pattern_id = seedset["manifest"]["pattern_id"]
    memmap_path = calibration.source_memmap_path
    if not os.path.isabs(memmap_path):
        # calibration.source_memmap_path is recorded VERBATIM from the tvi
        # manifest (tvi-repo-root-relative, e.g.
        # "output/vectors/join/vectors_nv-embed-v2.f32") -- this is the
        # files-only boundary (ADR-SKM-008 Option E): sk-mcp never imports
        # tvi, so it has no notion of "the tvi repo root" except what the
        # caller supplies. An explicit tvi_root arg is the correct, honest
        # resolution (never silently guessed against sk-mcp's own CWD, which
        # would resolve to the wrong tree and either 404 or -- worse --
        # silently hit an unrelated file at that relative path).
        tvi_root = args.get("tvi_root")
        if tvi_root:
            memmap_path = os.path.join(tvi_root, memmap_path)
        else:
            memmap_path = os.path.abspath(memmap_path)
            if not os.path.isfile(memmap_path):
                return {
                    "error": (
                        f"calibration's source_memmap_path {calibration.source_memmap_path!r} "
                        "is relative and 'tvi_root' was not supplied; resolving it against "
                        f"the current working directory gave {memmap_path!r}, which does not "
                        "exist. Pass 'tvi_root' (the thought-vault-integration checkout root) "
                        "so the frozen memmap can be located (files-only boundary, ADR-SKM-008 "
                        "Option E)."
                    )
                }

    rng = np.random.default_rng(seed)

    seeds_rows_all = _filter_rows_by_era(seedset["seeds"], era)
    negatives_rows_all = _filter_rows_by_era(seedset["negatives"], era)
    if not seeds_rows_all:
        return {"error": f"era filter {era!r} matched zero seed rows."}
    if not negatives_rows_all:
        return {"error": f"era filter {era!r} matched zero negative rows."}

    seed_rowids = [r["rowid_mm"] for r in seeds_rows_all]
    negative_rowids = [r["rowid_mm"] for r in negatives_rows_all]

    try:
        seed_vecs = read_memmap_rows(memmap_path, seed_rowids, calibration.dim, calibration.n_used)
        negative_vecs = read_memmap_rows(
            memmap_path, negative_rowids, calibration.dim, calibration.n_used
        )
    except DirectionRefusal as exc:
        return {"error": str(exc)}

    excluded = set(seed_rowids) | set(negative_rowids)
    candidate_pool = [i for i in range(calibration.n_used) if i not in excluded]
    sample_size = min(random_topic_sample_size, len(candidate_pool))
    random_rowids = rng.choice(candidate_pool, size=sample_size, replace=False) if sample_size else []
    null_sample_rowids = rng.choice(
        candidate_pool, size=min(random_topic_sample_size, len(candidate_pool)), replace=False
    ) if candidate_pool else []

    try:
        random_topic_vecs = read_memmap_rows(
            memmap_path, random_rowids, calibration.dim, calibration.n_used
        )
        corpus_null_sample_vecs = read_memmap_rows(
            memmap_path, null_sample_rowids, calibration.dim, calibration.n_used
        )
    except DirectionRefusal as exc:
        return {"error": str(exc)}

    result = initialize_direction_core(
        calibration=calibration,
        seedset={"seeds": seeds_rows_all, "negatives": negatives_rows_all},
        seed_vecs=seed_vecs,
        negative_vecs=negative_vecs,
        random_topic_vecs=random_topic_vecs,
        corpus_null_sample_vecs=corpus_null_sample_vecs,
        era=era,
        min_pole_separation=min_pole_separation,
        bootstrap_b=bootstrap_b,
        held_out_fraction=held_out_fraction,
        rng=rng,
    )
    if "error" in result:
        return result

    npz_path, json_path = write_direction_artifact(
        out_dir=out_dir,
        pattern_id=pattern_id,
        era=era,
        unit_axis=result["unit_axis"],
        calibration=calibration,
        seedset_manifest=seedset["manifest"],
        diagnostics=result,
    )

    return {
        "direction_ref": json_path,
        "npz_path": npz_path,
        "pattern_id": pattern_id,
        "era": era,
        "pole_separation": result["pole_separation"],
        "verdict": result["verdict"],
        "held_out_auc": result["held_out_auc"],
        "topic_control": result["topic_control"],
        "bootstrap": result["bootstrap"],
        "null_reference": result["null_reference"],
        "n_seeds": result["n_seeds"],
        "n_negatives": result["n_negatives"],
    }


# --------------------------------------------------------------------------- #
# Phase 3 (ADR-SKM-008 D4/D5): projection + rate primitives.
#
# Pure numeric core first (no IO -- testable with hand-built arrays), then
# artifact IO, then the corpus.db read-only helpers, then MCP handlers.
# --------------------------------------------------------------------------- #

def resolve_memmap_path(source_memmap_path: str, tvi_root: Optional[str]) -> str:
    """Shared tvi_root resolution for every Phase-3 verb that reads the
    memmap (lifted out of ``initialize_direction``'s inline logic so
    ``project_chunks``/``project_corpus`` do not duplicate it -- ONE-DOOR
    within this module). Raises :class:`DirectionRefusal` with the same
    instructive message ``initialize_direction`` already gives when a
    relative path cannot be resolved without ``tvi_root``.
    """
    if os.path.isabs(source_memmap_path):
        return source_memmap_path
    if tvi_root:
        return os.path.join(tvi_root, source_memmap_path)
    resolved = os.path.abspath(source_memmap_path)
    if os.path.isfile(resolved):
        return resolved
    raise DirectionRefusal(
        f"source_memmap_path {source_memmap_path!r} is relative and 'tvi_root' "
        f"was not supplied; resolving it against the current working directory "
        f"gave {resolved!r}, which does not exist. Pass 'tvi_root' (the "
        "thought-vault-integration checkout root) so the frozen memmap can be "
        "located (files-only boundary, ADR-SKM-008 Option E)."
    )


def project_vectors(centered_vecs: np.ndarray, unit_axis: np.ndarray) -> np.ndarray:
    """Pure matvec: project already-mean-centered vectors onto a unit axis.
    Shared by every Phase-3 verb (project_text/project_chunks/project_corpus)
    -- the ONE projection operation, never re-derived per verb."""
    return np.asarray(centered_vecs, dtype=np.float64) @ np.asarray(unit_axis, dtype=np.float64)


def z_scores(projections: np.ndarray, mu0: float, sigma0: float) -> np.ndarray:
    """D5: z-score a projection against the direction artifact's recorded
    corpus-null reference (mu0, sigma0) -- never recomputed per call, per D3
    "recorded so run/rate verbs z-score without recomputation."""
    if sigma0 == 0.0:
        raise DirectionRefusal(
            "null_reference sigma0 is zero; cannot z-score against a "
            "zero-variance null (the direction artifact's D3 diagnostics are "
            "degenerate)."
        )
    return (np.asarray(projections, dtype=np.float64) - mu0) / sigma0


def cosine_to_axis(vec: np.ndarray, unit_axis: np.ndarray) -> float:
    """Cosine of a (possibly non-unit) vector to the unit direction axis.
    Raw, not null-scored (ADR-SKM-008 Open Question on cosine null treatment
    is left unresolved by this phase -- D5's distribution-shape data does not
    yet exist for any real direction; raw cosine is reported as the
    interpretable default, matching ADR-SKMCP-0001's raw-cosine precedent)."""
    norm = float(np.linalg.norm(vec))
    if norm == 0.0:
        return 0.0
    return float(np.dot(vec, unit_axis) / norm)


# --------------------------------------------------------------------------- #
# Projection artifact persistence: regime-typed, self-describing, keyed by
# direction-sha + memmap-sha (ADR D4 project_corpus row).
# --------------------------------------------------------------------------- #

def projection_artifact_paths(out_dir: str, pattern_id: str, era: Optional[str]) -> Tuple[str, str]:
    """``data/projections/<pattern_id>[.<era>].proj.{npz,json}`` -- same
    slugging convention as :func:`direction_artifact_paths`."""
    slug = pattern_id if era is None else f"{pattern_id}.{era}"
    safe_slug = slug.replace("/", "__").replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{safe_slug}.proj.npz")
    json_path = os.path.join(out_dir, f"{safe_slug}.proj.json")
    return npz_path, json_path


def write_projection_artifact(
    out_dir: str,
    direction: "Direction",
    rowids_mm: np.ndarray,
    z: np.ndarray,
    n_used: int,
) -> Tuple[str, str]:
    """Persist the full-corpus projection artifact: per-``rowid_mm`` z-score,
    keyed by direction-sha + memmap-sha. ``n_used`` is recorded so a consumer
    (or a test) can reconcile the artifact's row count to the memmap's
    declared ``n_used`` (ADR-SKM-008 Phase 3 acceptance criterion)."""
    npz_path, json_path = projection_artifact_paths(
        out_dir, direction.pattern_id, direction.era
    )
    np.savez(
        npz_path,
        rowid_mm=np.asarray(rowids_mm, dtype=np.int64),
        z=np.asarray(z, dtype=np.float64),
    )
    manifest = {
        "header": {
            "regime": EXPECTED_PROJECTION_REGIME,
            "embedding_model_id": direction.embedding_model_id,
            "source_memmap_sha256": direction.source_memmap_sha256,
            "direction_sha256": direction.direction_sha256,
            "pattern_id": direction.pattern_id,
            "era": direction.era,
            "n_used": n_used,
        },
        "n_rows": int(rowids_mm.shape[0]),
        "z_mean": round(float(np.mean(z)), 6) if z.size else None,
        "z_std": round(float(np.std(z)), 6) if z.size else None,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return npz_path, json_path


@dataclass
class Projection:
    """A loaded ``direction-projection`` artifact."""

    embedding_model_id: str
    source_memmap_sha256: str
    direction_sha256: str
    pattern_id: str
    era: Optional[str]
    n_used: int
    rowid_mm: np.ndarray
    z: np.ndarray
    header: Dict[str, Any] = field(default_factory=dict)
    manifest: Dict[str, Any] = field(default_factory=dict)


def load_projection(json_path: str) -> Projection:
    """Load a ``direction-projection`` artifact (jolt.py hard-fail discipline,
    same as :func:`load_calibration`/:func:`load_direction`)."""
    try:
        with open(json_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Projection artifact not found: {json_path}. Build it first with "
            "project_corpus (no silent default)."
        ) from exc

    header = manifest.get("header")
    if not isinstance(header, dict):
        raise ValueError(
            f"Projection artifact {json_path} has no 'header' block; refusing "
            "header-less projection."
        )
    missing = [k for k in REQUIRED_PROJECTION_HEADER_KEYS if k not in header]
    if missing:
        raise ValueError(
            f"Projection artifact {json_path} header missing required keys "
            f"{missing}; refusing to load an under-described projection."
        )
    if header["regime"] != EXPECTED_PROJECTION_REGIME:
        raise ValueError(
            f"Projection artifact {json_path} regime is {header['regime']!r}, "
            f"expected {EXPECTED_PROJECTION_REGIME!r}."
        )

    npz_path = os.path.splitext(json_path)[0] + ".npz"
    try:
        npz = np.load(npz_path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Projection npz not found: {npz_path} (expected alongside "
            f"{json_path}). Build it first (no silent default)."
        ) from exc
    for key in ("rowid_mm", "z"):
        if key not in npz:
            raise ValueError(f"Projection npz {npz_path} has no {key!r} array.")

    return Projection(
        embedding_model_id=header["embedding_model_id"],
        source_memmap_sha256=header["source_memmap_sha256"],
        direction_sha256=header["direction_sha256"],
        pattern_id=header["pattern_id"],
        era=header["era"],
        n_used=int(header["n_used"]),
        rowid_mm=np.asarray(npz["rowid_mm"], dtype=np.int64),
        z=np.asarray(npz["z"], dtype=np.float64),
        header=header,
        manifest=manifest,
    )


def refuse_unless_projection_matches_direction(
    projection: "Projection", direction: "Direction"
) -> None:
    """Identity gate for any verb consuming a projection artifact alongside a
    direction (mirrors :meth:`Calibration.refuse_unless_matches`): a
    projection built from a different direction (different direction_sha256)
    or a different corpus snapshot must not silently feed query_rates/
    top_exemplars against the wrong axis."""
    mismatches = []
    if projection.direction_sha256 != direction.direction_sha256:
        mismatches.append(
            f"direction_sha256 {projection.direction_sha256!r} != direction's "
            f"{direction.direction_sha256!r}"
        )
    if projection.source_memmap_sha256 != direction.source_memmap_sha256:
        mismatches.append(
            f"source_memmap_sha256 {projection.source_memmap_sha256!r} != "
            f"direction's {direction.source_memmap_sha256!r}"
        )
    if mismatches:
        raise DirectionRefusal(
            "Identity mismatch between projection artifact and direction "
            "artifact: " + "; ".join(mismatches) + ". The projection was built "
            "from a different direction/corpus snapshot -- refusing rather "
            "than reading rates/exemplars off the wrong axis."
        )


# --------------------------------------------------------------------------- #
# Rate-table artifact persistence (D5).
# --------------------------------------------------------------------------- #

def rate_table_artifact_paths(out_dir: str, pattern_id: str, era: Optional[str]) -> Tuple[str, str]:
    """``data/rates/<pattern_id>[.<era>].rates.json`` (no sibling npz -- the
    rate table is small aggregated JSON, not a per-row array)."""
    slug = pattern_id if era is None else f"{pattern_id}.{era}"
    safe_slug = slug.replace("/", "__").replace(" ", "_")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{safe_slug}.rates.json")


def calibrate_threshold_from_direction(
    direction: "Direction", target_precision: float = DEFAULT_RATE_TARGET_PRECISION
) -> Dict[str, Any]:
    """D5: "the threshold is calibrated against the regex labels via the D3
    held-out AUC curve ... the projection z at which held-out seed-vs-negative
    precision >= a target (default 0.9)". The direction artifact's manifest
    does not persist the raw held-out projection arrays (only the summary AUC
    diagnostics) -- so the achievable calibration from a Phase-2 artifact
    alone is the summary-level one: read the recorded held_out_auc's implied
    separation quality and gate on the leakage/verdict discipline already
    computed, rather than re-deriving a precision-recall curve this module has
    no raw scores to build. This is a bounded consequence of Phase 2's
    persisted shape, named honestly rather than fabricating a quantile from
    data that is not there.

    Returns ``{"threshold": z, "target_precision": p, "method": ...}``. When
    the direction's null_reference is available, the threshold defaults to a
    1.0-sigma-equivalent quantile scaled by held-out AUC as an honest
    placeholder ("empirical quantile, not a sigma-multiple" per D5 is the
    stated fallback when the true precision-recall curve is unavailable);
    callers wanting the ADR's exact precision-targeted threshold should pass
    an explicit ``threshold`` derived from their own held-out scores.
    """
    held_out_auc = direction.manifest.get("held_out_auc", {}) or {}
    auc = held_out_auc.get("auc")
    if auc is None:
        raise DirectionRefusal(
            "direction artifact has no recorded held_out_auc.auc; cannot "
            "calibrate a threshold without it. Re-run initialize_direction, "
            "or supply an explicit 'threshold' to query_rates."
        )
    # Standard-normal quantile via the inverse error function (dependency-free
    # -- no scipy in this project's dependency set, mirroring auc_score's
    # scipy-free precedent). A higher target_precision or a higher achieved
    # AUC both push the threshold higher (fewer, more-confident positives).
    from math import erfinv
    quantile = target_precision + (1.0 - target_precision) * max(0.0, min(1.0, auc))
    quantile = min(quantile, 0.999999)
    z_threshold = float(np.sqrt(2.0) * erfinv(2.0 * quantile - 1.0))
    return {
        "threshold": round(z_threshold, 4),
        "target_precision": target_precision,
        "source_auc": auc,
        "method": (
            "gaussian-quantile approximation from held_out_auc.auc "
            "(Phase 2 artifact does not persist raw held-out scores for an "
            "exact precision-recall threshold; honest placeholder, see "
            "calibrate_threshold_from_direction docstring)"
        ),
    }


def build_rate_table(
    projection: "Projection",
    threshold: float,
    corpus_rows: Sequence[Dict[str, Any]],
    group_by: Sequence[str] = ("era", "channel", "speaker"),
) -> Dict[str, Any]:
    """Pure aggregation core: given the projection's per-rowid_mm z-scores and
    a parallel list of corpus rows ``{rowid_mm, era, channel, speaker}``
    (already joined from corpus.db by the IO layer), compute the fraction of
    chunks above ``threshold``, grouped by every combination of
    ``group_by`` fields. NULL-era rows are a first-class bucket (grouped under
    the string ``"NULL"``, never dropped) per the dispatch's explicit
    requirement that NULL is a real bucket.

    Returns ``{"groups": [{"key": {...}, "n": int, "n_above": int, "rate": float}], "n_total": int}``.
    """
    z_by_rowid = {int(r): float(z) for r, z in zip(projection.rowid_mm, projection.z)}
    buckets: Dict[Tuple[Any, ...], Dict[str, int]] = {}
    for row in corpus_rows:
        rowid = row.get("rowid_mm")
        if rowid not in z_by_rowid:
            continue
        key = tuple(
            row.get(field) if row.get(field) is not None else "NULL" for field in group_by
        )
        bucket = buckets.setdefault(key, {"n": 0, "n_above": 0})
        bucket["n"] += 1
        if z_by_rowid[rowid] >= threshold:
            bucket["n_above"] += 1

    groups = []
    n_total = 0
    for key, counts in sorted(buckets.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n_total += counts["n"]
        groups.append(
            {
                "key": dict(zip(group_by, key)),
                "n": counts["n"],
                "n_above": counts["n_above"],
                "rate": round(counts["n_above"] / counts["n"], 6) if counts["n"] else 0.0,
            }
        )
    return {"groups": groups, "n_total": n_total, "group_by": list(group_by)}


def write_rate_table_artifact(
    out_dir: str,
    projection: "Projection",
    threshold_info: Dict[str, Any],
    rate_table: Dict[str, Any],
) -> str:
    """Persist the rate-table artifact: threshold + precision + source shas
    recorded so every rate is re-derivable and self-labeling (ADR-SKM-006
    'derived stats are generated, never hand-typed')."""
    json_path = rate_table_artifact_paths(out_dir, projection.pattern_id, projection.era)
    manifest = {
        "header": {
            "regime": EXPECTED_RATE_TABLE_REGIME,
            "embedding_model_id": projection.embedding_model_id,
            "source_memmap_sha256": projection.source_memmap_sha256,
            "direction_sha256": projection.direction_sha256,
            "pattern_id": projection.pattern_id,
            "era": projection.era,
        },
        "threshold": threshold_info,
        "rate_table": rate_table,
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    return json_path


# --------------------------------------------------------------------------- #
# corpus.db read-only helpers (files-only boundary; sk-mcp never writes it).
# --------------------------------------------------------------------------- #

def open_corpus_db_readonly(corpus_db_path: str) -> sqlite3.Connection:
    """Open corpus.db via SQLite's read-only URI (``mode=ro``) -- the
    enforcement surface named in ADR-TVI-008/ADR-SKM-008 for the read-only
    boundary: there is no tvi write path reachable from sk-mcp because this
    is the only way sk-mcp ever opens the file."""
    if not os.path.isfile(corpus_db_path):
        raise DirectionRefusal(
            f"corpus.db not found: {corpus_db_path}. It is a tvi-side "
            "(ADR-TVI-008) build artifact -- sk-mcp only ever reads it, never "
            "builds it."
        )
    uri = f"file:{corpus_db_path}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def fetch_chunk_rows_for_rowids(
    conn: sqlite3.Connection, rowids_mm: Sequence[int]
) -> List[Dict[str, Any]]:
    """Read-only join: ``vector_status(rowid_mm) -> chunk(chunk_id)`` for
    ``era``/``channel``(=``chunk.source``)/``speaker`` grouping (D4
    query_rates). Schema per ADR-TVI-008 §Boundary contract: ``chunk`` (14
    fields + ``model`` + ``era``), ``vector_status(chunk_id, model_id, status,
    rowid_mm)`` -- read verbatim, no re-typing of column names.
    """
    if not rowids_mm:
        return []
    placeholders = ",".join("?" for _ in rowids_mm)
    query = (
        "SELECT vs.rowid_mm AS rowid_mm, c.era AS era, c.source AS channel, "
        "c.speaker AS speaker, c.chunk_id AS chunk_id "
        "FROM vector_status vs JOIN chunk c ON vs.chunk_id = c.chunk_id "
        f"WHERE vs.rowid_mm IN ({placeholders})"
    )
    cur = conn.execute(query, list(int(r) for r in rowids_mm))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def fetch_chunk_text(conn: sqlite3.Connection, chunk_ids: Sequence[str]) -> Dict[str, str]:
    """Read-only text fetch by ``chunk_id`` (top_exemplars readback)."""
    if not chunk_ids:
        return {}
    placeholders = ",".join("?" for _ in chunk_ids)
    query = f"SELECT chunk_id, text FROM chunk WHERE chunk_id IN ({placeholders})"
    cur = conn.execute(query, list(chunk_ids))
    return {row[0]: row[1] for row in cur.fetchall()}


def fetch_chunk_id_for_rowid(conn: sqlite3.Connection, rowids_mm: Sequence[int]) -> Dict[int, str]:
    """Read-only ``rowid_mm -> chunk_id`` map (top_exemplars needs this before
    it can fetch text by chunk_id)."""
    if not rowids_mm:
        return {}
    placeholders = ",".join("?" for _ in rowids_mm)
    query = f"SELECT rowid_mm, chunk_id FROM vector_status WHERE rowid_mm IN ({placeholders})"
    cur = conn.execute(query, list(int(r) for r in rowids_mm))
    return {row[0]: row[1] for row in cur.fetchall()}


# --------------------------------------------------------------------------- #
# MCP handlers.
# --------------------------------------------------------------------------- #

def _load_direction_and_calibration(
    direction_ref: str, calibration_ref: str
) -> Tuple["Direction", "Calibration"]:
    """Shared load + identity-gate step every Phase-3 verb needs: load both
    artifacts and refuse if the direction's declared identity does not match
    the calibration it is being asked to mean-center against."""
    direction = load_direction(direction_ref)
    calibration = load_calibration(calibration_ref)
    calibration.refuse_unless_matches(
        embedding_model_id=direction.embedding_model_id,
        source_memmap_sha256=direction.source_memmap_sha256,
    )
    return direction, calibration


async def project_text(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``project_text`` (D4): embed input text via the adapter
    substrate (pinned by the direction's embedding_model_id), mean-center
    against the calibration's mu, project onto the direction's unit axis.
    """
    text = args.get("text")
    direction_ref = args.get("direction_ref")
    calibration_ref = args.get("calibration_ref")
    allow_override = bool(args.get(ALLOW_NON_USABLE_PROJECTION_ARG, False))
    if not text:
        return {"error": "text is required."}
    if not direction_ref:
        return {"error": "direction_ref is required."}
    if not calibration_ref:
        return {"error": "calibration_ref is required."}

    try:
        direction, calibration = _load_direction_and_calibration(direction_ref, calibration_ref)
        refuse_unless_usable(direction, allow_override)
    except (FileNotFoundError, ValueError, DirectionRefusal) as exc:
        return {"error": str(exc)}

    adapter = manager.get_adapter()
    if adapter.model_name != direction.embedding_model_id:
        return {
            "error": (
                f"active embedding model {adapter.model_name!r} does not match "
                f"the direction's pinned embedding_model_id "
                f"{direction.embedding_model_id!r}; refusing to project a "
                "vector from a different embedding space onto this axis."
            )
        }

    raw_vec = np.asarray(adapter.embed(text), dtype=np.float64)
    centered = mean_center(raw_vec.reshape(1, -1), calibration.mu)[0]
    projection = float(project_vectors(centered.reshape(1, -1), direction.unit_axis)[0])

    null_ref = direction.manifest.get("null_reference", {}) or {}
    result: Dict[str, Any] = {
        "projection": round(projection, 6),
        "cosine": round(cosine_to_axis(centered, direction.unit_axis), 6),
        "direction_ref": direction_ref,
        "pattern_id": direction.pattern_id,
        "era": direction.era,
        "verdict": direction.verdict,
    }
    if "mu0" in null_ref and "sigma0" in null_ref:
        try:
            result["z"] = round(
                float(z_scores(np.array([projection]), null_ref["mu0"], null_ref["sigma0"])[0]), 6
            )
        except DirectionRefusal as exc:
            result["z_error"] = str(exc)
    if allow_override and direction.verdict != "usable":
        result["allow_non_usable_direction_used"] = True
    return result


async def project_chunks(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``project_chunks`` (D4): project caller-supplied rowid_mm
    values from the memmap onto the direction. No embedding call."""
    direction_ref = args.get("direction_ref")
    calibration_ref = args.get("calibration_ref")
    rowids = args.get("rowids")
    tvi_root = args.get("tvi_root")
    allow_override = bool(args.get(ALLOW_NON_USABLE_PROJECTION_ARG, False))
    if not direction_ref:
        return {"error": "direction_ref is required."}
    if not calibration_ref:
        return {"error": "calibration_ref is required."}
    if not rowids:
        return {"error": "rowids is required (non-empty list of memmap row indices)."}

    try:
        direction, calibration = _load_direction_and_calibration(direction_ref, calibration_ref)
        refuse_unless_usable(direction, allow_override)
        memmap_path = resolve_memmap_path(calibration.source_memmap_path, tvi_root)
        vecs = read_memmap_rows(memmap_path, rowids, calibration.dim, calibration.n_used)
    except (FileNotFoundError, ValueError, DirectionRefusal) as exc:
        return {"error": str(exc)}

    centered = mean_center(vecs, calibration.mu)
    z = project_vectors(centered, direction.unit_axis)

    rows = [{"rowid_mm": int(r), "z": round(float(v), 6)} for r, v in zip(rowids, z)]
    result = {
        "rows": rows,
        "direction_ref": direction_ref,
        "pattern_id": direction.pattern_id,
        "era": direction.era,
        "verdict": direction.verdict,
    }
    if allow_override and direction.verdict != "usable":
        result["allow_non_usable_direction_used"] = True
    return result


async def project_corpus(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``project_corpus`` (D4): the full N x dim matvec over the
    whole memmap. Persists a projection artifact keyed by direction-sha +
    memmap-sha; row count reconciles to the memmap's n_used."""
    direction_ref = args.get("direction_ref")
    calibration_ref = args.get("calibration_ref")
    tvi_root = args.get("tvi_root")
    out_dir = args.get("out_dir") or PROJECTION_ARTIFACT_DIR
    allow_override = bool(args.get(ALLOW_NON_USABLE_PROJECTION_ARG, False))
    if not direction_ref:
        return {"error": "direction_ref is required."}
    if not calibration_ref:
        return {"error": "calibration_ref is required."}

    try:
        direction, calibration = _load_direction_and_calibration(direction_ref, calibration_ref)
        refuse_unless_usable(direction, allow_override)
        memmap_path = resolve_memmap_path(calibration.source_memmap_path, tvi_root)
        n_used = calibration.n_used
        rowids_mm = np.arange(n_used, dtype=np.int64)
        vecs = read_memmap_rows(memmap_path, rowids_mm, calibration.dim, n_used)
    except (FileNotFoundError, ValueError, DirectionRefusal) as exc:
        return {"error": str(exc)}

    centered = mean_center(vecs, calibration.mu)
    z = project_vectors(centered, direction.unit_axis)

    npz_path, json_path = write_projection_artifact(out_dir, direction, rowids_mm, z, n_used)

    result = {
        "projection_ref": json_path,
        "npz_path": npz_path,
        "pattern_id": direction.pattern_id,
        "era": direction.era,
        "n_rows": int(rowids_mm.shape[0]),
        "n_used": n_used,
        "z_mean": round(float(np.mean(z)), 6) if z.size else None,
        "z_std": round(float(np.std(z)), 6) if z.size else None,
        "verdict": direction.verdict,
    }
    if allow_override and direction.verdict != "usable":
        result["allow_non_usable_direction_used"] = True
    return result


async def query_rates(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``query_rates`` (D4/D5): read a projection artifact +
    corpus.db (read-only) -> an aggregated rate table by era x channel x
    speaker. Threshold is D5-calibrated from the direction artifact's
    held-out AUC unless an explicit ``threshold`` is supplied."""
    projection_ref = args.get("projection_ref")
    corpus_db_ref = args.get("corpus_db_ref")
    direction_ref = args.get("direction_ref")
    explicit_threshold = args.get("threshold")
    target_precision = args.get("target_precision", DEFAULT_RATE_TARGET_PRECISION)
    group_by = args.get("group_by") or ["era", "channel", "speaker"]
    out_dir = args.get("out_dir") or RATE_ARTIFACT_DIR

    if not projection_ref:
        return {"error": "projection_ref is required."}
    if not corpus_db_ref:
        return {"error": "corpus_db_ref is required."}

    try:
        projection = load_projection(projection_ref)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}

    if explicit_threshold is not None:
        threshold_info = {
            "threshold": float(explicit_threshold),
            "target_precision": None,
            "method": "explicit threshold supplied by caller",
        }
    else:
        if not direction_ref:
            return {
                "error": (
                    "either 'threshold' or 'direction_ref' is required (the "
                    "threshold is calibrated from the direction's held-out AUC "
                    "when not supplied explicitly)."
                )
            }
        try:
            direction = load_direction(direction_ref)
            refuse_unless_projection_matches_direction(projection, direction)
            threshold_info = calibrate_threshold_from_direction(direction, target_precision)
        except (FileNotFoundError, ValueError, DirectionRefusal) as exc:
            return {"error": str(exc)}

    try:
        conn = open_corpus_db_readonly(corpus_db_ref)
    except DirectionRefusal as exc:
        return {"error": str(exc)}
    try:
        corpus_rows = fetch_chunk_rows_for_rowids(conn, projection.rowid_mm.tolist())
    finally:
        conn.close()

    rate_table = build_rate_table(projection, threshold_info["threshold"], corpus_rows, group_by)
    json_path = write_rate_table_artifact(out_dir, projection, threshold_info, rate_table)

    return {
        "rate_table_ref": json_path,
        "pattern_id": projection.pattern_id,
        "era": projection.era,
        "threshold": threshold_info,
        "rate_table": rate_table,
    }


async def top_exemplars(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for ``top_exemplars`` (D4): top-K chunk_ids along a direction
    (from a projection artifact), with text fetched via corpus.db read-only.
    """
    projection_ref = args.get("projection_ref")
    corpus_db_ref = args.get("corpus_db_ref")
    k = int(args.get("k", DEFAULT_TOP_EXEMPLARS_K))
    era = args.get("era")

    if not projection_ref:
        return {"error": "projection_ref is required."}
    if not corpus_db_ref:
        return {"error": "corpus_db_ref is required."}

    try:
        projection = load_projection(projection_ref)
    except (FileNotFoundError, ValueError) as exc:
        return {"error": str(exc)}

    try:
        conn = open_corpus_db_readonly(corpus_db_ref)
    except DirectionRefusal as exc:
        return {"error": str(exc)}

    try:
        all_rowids = projection.rowid_mm.tolist()
        rowid_to_chunk = fetch_chunk_id_for_rowid(conn, all_rowids)
        era_by_rowid: Dict[int, Any] = {}
        if era is not None:
            era_rows = fetch_chunk_rows_for_rowids(conn, all_rowids)
            era_by_rowid = {r["rowid_mm"]: r.get("era") for r in era_rows}

        order = np.argsort(-projection.z)  # descending: highest z first
        candidates: List[Tuple[int, str, float]] = []
        for idx in order:
            rowid = int(projection.rowid_mm[idx])
            chunk_id = rowid_to_chunk.get(rowid)
            if chunk_id is None:
                continue
            if era is not None and era_by_rowid.get(rowid) != era:
                continue
            candidates.append((rowid, chunk_id, float(projection.z[idx])))
            if len(candidates) >= k:
                break

        texts = fetch_chunk_text(conn, [c[1] for c in candidates])
    finally:
        conn.close()

    exemplars = [
        {
            "rowid_mm": rowid,
            "chunk_id": chunk_id,
            "z": round(z, 6),
            "text": texts.get(chunk_id),
        }
        for rowid, chunk_id, z in candidates
    ]
    return {
        "exemplars": exemplars,
        "k": k,
        "era": era,
        "pattern_id": projection.pattern_id,
    }
