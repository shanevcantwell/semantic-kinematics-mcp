"""
Axis-alignment command module: referential projection analysis.

Where trajectory.py measures *reflexive* geometry (inter-step angles/curvature,
which degenerate toward orthogonality in high-dimensional space), this module
measures *referential* geometry: how strongly a passage marches along a
caller-specified semantic axis, and whether that march is significant relative
to a background-corpus null.

See docs/ADRs/proposed/ADR-SKM-0007-referential-axis-alignment.md.

Tools:
- analyze_axis_alignment: project a passage onto an anchor-defined axis and
  z-score the projection against a cached background null.

The numerics live in pure functions (build_axis / null_stats / alignment_core)
so they are testable without spaCy or a real embedding backend.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from mcp.types import Tool

from semantic_kinematics.mcp.state_manager import StateManager

# Default floor for pole separation on (L2-normalized) embeddings. Below this
# the positive/negative poles are effectively collinear and the axis is noise.
DEFAULT_MIN_POLE_SEPARATION = 0.05


# --------------------------------------------------------------------------- #
# Pure numerics (no IO, no spaCy, no adapter) -- the testable core.
# --------------------------------------------------------------------------- #

def build_axis(
    pos_vecs: np.ndarray,
    neg_pole: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Build a unit reference axis from averaged positive exemplars and a negative
    pole.

    Args:
        pos_vecs: (n_pos, d) positive-anchor exemplar embeddings.
        neg_pole: (d,) the negative pole -- either averaged negative exemplars
            or the background-null mean (see Decision 6 in the ADR).

    Returns:
        (unit_axis (d,), pole_separation) where pole_separation is the raw
        ||pos_mean - neg_pole|| before normalization.
    """
    pos_mean = np.asarray(pos_vecs, dtype=np.float64).mean(axis=0)
    raw = pos_mean - np.asarray(neg_pole, dtype=np.float64)
    separation = float(np.linalg.norm(raw))
    if separation == 0.0:
        # Degenerate: poles coincide. Caller gates on separation; return a zero
        # axis so any accidental use projects to zero rather than NaN.
        return np.zeros_like(raw), 0.0
    return raw / separation, separation


def null_stats(null_embeddings: np.ndarray, unit_axis: np.ndarray) -> Tuple[float, float]:
    """Mean and std of the background corpus projected onto the axis."""
    proj = np.asarray(null_embeddings, dtype=np.float64) @ np.asarray(unit_axis, dtype=np.float64)
    return float(proj.mean()), float(proj.std())


def alignment_core(
    sentence_embeddings: np.ndarray,
    pos_vecs: np.ndarray,
    neg_vecs: Optional[np.ndarray],
    null_embeddings: np.ndarray,
    min_pole_separation: float = DEFAULT_MIN_POLE_SEPARATION,
) -> Dict[str, Any]:
    """
    Compute the axis-alignment profile.

    When neg_vecs is None the negative pole is the background-null mean (so the
    axis points from the cone center toward the positive concept, de-meaning the
    anisotropy in one move).

    Returns a result dict, or an {"error": ...} dict when a precondition fails
    (too few sentences, underdetermined axis, degenerate null).
    """
    sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float64)
    null_embeddings = np.asarray(null_embeddings, dtype=np.float64)

    if sentence_embeddings.shape[0] < 2:
        return {"error": "Need at least 2 sentences to measure an axis march."}
    if null_embeddings.shape[0] < 2:
        return {"error": "Background null needs at least 2 embeddings."}

    null_mean = null_embeddings.mean(axis=0)
    neg_pole = np.asarray(neg_vecs, dtype=np.float64).mean(axis=0) if neg_vecs is not None else null_mean

    unit_axis, pole_separation = build_axis(pos_vecs, neg_pole)
    if pole_separation < min_pole_separation:
        return {
            "error": "axis underdetermined",
            "detail": (
                f"pole separation {pole_separation:.4f} < minimum "
                f"{min_pole_separation:.4f}; the anchors embed too close to "
                "define a stable axis."
            ),
            "pole_separation": pole_separation,
        }

    mu0, sigma0 = null_stats(null_embeddings, unit_axis)
    if sigma0 == 0.0:
        return {"error": "Background null has zero variance along this axis."}

    # Position trace: where each sentence sits on the axis, in sigma units.
    proj = sentence_embeddings @ unit_axis
    zscores = (proj - mu0) / sigma0

    # Step projections s_i = (e_{i+1} - e_i) . axis are exactly diffs of proj.
    steps = np.diff(proj)
    total_step = float(np.sum(np.abs(steps)))
    net_step = float(np.sum(steps))
    axis_straightness = abs(net_step) / total_step if total_step > 0 else 0.0

    return {
        "n_sentences": int(sentence_embeddings.shape[0]),
        "position_zscores": [round(float(z), 4) for z in zscores],
        "axis_drift": round(float(zscores[-1] - zscores[0]), 4),  # net march, sigma units
        "axis_straightness": round(axis_straightness, 4),
        "mean_zscore": round(float(zscores.mean()), 4),
        "pole_separation": round(pole_separation, 4),
        "null_count": int(null_embeddings.shape[0]),
    }


# --------------------------------------------------------------------------- #
# Background null cache IO.
# --------------------------------------------------------------------------- #

def load_null_cache(manifest_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load a background null cache from a manifest JSON.

    The manifest carries {model_name, dimensions, count, embeddings_path,
    source}. The embeddings .npy lives alongside (path resolved relative to the
    manifest dir when not absolute).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)
    emb_path = manifest["embeddings_path"]
    if not os.path.isabs(emb_path):
        emb_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)), emb_path)
    embeddings = np.load(emb_path)
    return embeddings, manifest


def build_null_cache(
    adapter,
    texts: List[str],
    out_npy: str,
    source: str = "",
) -> Dict[str, Any]:
    """
    Embed a background corpus once and persist embeddings + manifest.

    Writes `out_npy` (the embeddings) and a sibling `<out_npy>.json` manifest
    keyed by adapter.model_name. Returns the manifest dict.
    """
    if not out_npy.endswith(".npy"):
        out_npy += ".npy"  # keep manifest's stored basename in sync with np.save
    embeddings = adapter.embed_batch(texts)
    embeddings = np.asarray(embeddings)
    os.makedirs(os.path.dirname(os.path.abspath(out_npy)) or ".", exist_ok=True)
    np.save(out_npy, embeddings)
    manifest = {
        "model_name": adapter.model_name,
        "dimensions": int(embeddings.shape[1]),
        "count": int(embeddings.shape[0]),
        "embeddings_path": os.path.basename(out_npy),
        "source": source,
    }
    manifest_path = out_npy + ".json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def _split_exemplars(raw: str) -> List[str]:
    """Newline-separated exemplars, stripped, empties dropped."""
    return [line.strip() for line in raw.splitlines() if line.strip()]


# --------------------------------------------------------------------------- #
# MCP tool surface.
# --------------------------------------------------------------------------- #

def get_tools() -> List[Tool]:
    """Return axis-alignment tool definitions."""
    return [
        Tool(
            name="analyze_axis_alignment",
            description=(
                "Measure how strongly a passage marches along a caller-defined "
                "semantic axis, z-scored against a background-corpus null. The "
                "axis is built from anchor exemplars (anchor_positive minus "
                "anchor_negative, or minus the null mean when no negative is "
                "given). Returns a per-sentence position trace in sigma units, "
                "the net axis drift, and axis-restricted straightness (1.0 = a "
                "disciplined straight-line march along the axis). z-scores are "
                "relative to the supplied background population."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Passage to analyze (needs 2+ sentences)",
                    },
                    "anchor_positive": {
                        "type": "string",
                        "description": (
                            "Positive pole. Newline-separate multiple exemplars; "
                            "they are averaged for a robust axis."
                        ),
                    },
                    "anchor_negative": {
                        "type": "string",
                        "description": (
                            "Optional negative pole (newline-separated exemplars). "
                            "If omitted, the background-null mean is used."
                        ),
                    },
                    "background_ref": {
                        "type": "string",
                        "description": (
                            "Path to a background null manifest JSON (built with "
                            "scripts/build_axis_null.py). Required: z-scores are "
                            "meaningless without it. Defaults to env "
                            "AXIS_NULL_MANIFEST."
                        ),
                    },
                    "min_pole_separation": {
                        "type": "number",
                        "description": "Axis-quality floor (default: 0.05).",
                        "default": DEFAULT_MIN_POLE_SEPARATION,
                    },
                    "include_sentences": {
                        "type": "boolean",
                        "description": "Echo sentence breakdown in output (default: false).",
                        "default": False,
                    },
                },
                "required": ["text", "anchor_positive"],
            },
        ),
    ]


async def analyze_axis_alignment(manager: StateManager, args: Dict[str, Any]) -> Dict[str, Any]:
    """Handler: tokenize, embed anchors + passage, load null, run alignment_core."""
    text = args.get("text", "")
    anchor_positive = args.get("anchor_positive", "")
    anchor_negative = args.get("anchor_negative")
    background_ref = args.get("background_ref") or os.environ.get("AXIS_NULL_MANIFEST")
    min_pole_separation = args.get("min_pole_separation", DEFAULT_MIN_POLE_SEPARATION)
    include_sentences = args.get("include_sentences", False)

    pos_exemplars = _split_exemplars(anchor_positive)
    if not pos_exemplars:
        return {"error": "anchor_positive must contain at least one exemplar."}
    if not background_ref:
        return {
            "error": (
                "background_ref (or AXIS_NULL_MANIFEST) is required; z-scores are "
                "meaningless without a background null."
            )
        }

    # Load null and verify it matches the active model's geometry.
    try:
        null_embeddings, manifest = load_null_cache(background_ref)
    except (OSError, KeyError, ValueError) as e:
        return {"error": f"Could not load background null from {background_ref!r}: {e}"}

    adapter = manager.get_adapter()
    if manifest.get("model_name") != adapter.model_name:
        return {
            "error": (
                f"Background null was built for model {manifest.get('model_name')!r} "
                f"but the active model is {adapter.model_name!r}; geometry differs. "
                "Rebuild the null for this backend."
            )
        }

    # Reuse the trajectory analyzer's spaCy sentence tokenizer.
    from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer
    analyzer = TrajectoryAnalyzer(manager)
    sentences = analyzer.tokenize_sentences(text)
    if len(sentences) < 2:
        return {"error": "Need at least 2 sentences to measure an axis march."}

    sentence_embeddings = analyzer.embed_sentences(sentences)
    pos_vecs = adapter.embed_batch(pos_exemplars)
    neg_exemplars = _split_exemplars(anchor_negative) if anchor_negative else []
    neg_vecs = adapter.embed_batch(neg_exemplars) if neg_exemplars else None

    result = alignment_core(
        sentence_embeddings,
        pos_vecs,
        neg_vecs,
        null_embeddings,
        min_pole_separation=min_pole_separation,
    )

    if "error" not in result:
        result["model_name"] = adapter.model_name
        result["background_ref"] = background_ref
        if include_sentences:
            result["sentences"] = sentences
    return result
