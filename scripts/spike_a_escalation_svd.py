#!/usr/bin/env python3
"""
Spike A — escalation-axis dimensionality probe.

Embeds the escalation anchor grid (data/anchors/escalation_grid.yaml), forms
(escalated - level) delta vectors per axis pair, and interrogates whether the
three hypothesized axes (tone, urgency, importance) are one collapsed manifold
or genuinely separable directions.

REPORTS NUMBERS ONLY. No go/no-go verdict — that is the caller's call.

Steps:
  1. Embed all 24 axis pairs (level + escalated = 48 texts) + 4 mood pairs (8 texts).
     Unit-normalize (L2) every embedding.
  2. Deltas d_i = embed(escalated_i) - embed(level_i), 24 axis + 4 held-out mood.
  3. Union SVD on the 24x768 stacked axis-delta matrix, RAW and MEAN-CENTERED:
     singular spectrum as raw values, fraction of variance, cumulative top-5.
  4. Per-axis separability: 3 pairwise cosines between normalized per-axis mean deltas.
  5. Within-axis coherence: mean pairwise cosine among each axis's 8 normalized deltas.
  6. Length-confound: corr(len-diff, ||d_i||) — char and token (via /tokenize) Pearson r.
  7. Mood projection: 4x3 cosine table of held-out mood deltas vs the 3 axis means.

Reuses semantic_kinematics.embeddings.lmstudio.LMStudioAdapter (OpenAI-compatible,
points at :8082 embeddinggemma; count_tokens via llama.cpp /tokenize).
"""

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Make repo importable when run from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter  # noqa: E402

DEFAULT_BASE_URL = "http://localhost:8082/v1"
DEFAULT_MODEL = "embeddinggemma-300M-F32.gguf"
DEFAULT_GRID = REPO_ROOT / "data" / "anchors" / "escalation_grid.yaml"
DEFAULT_OUT = Path("/tmp/sk_spike_a/results.json")

AXES = ("tone", "urgency", "importance")


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def svd_spectrum(matrix: np.ndarray, top: int = 5):
    """Return raw singular values, fraction-of-variance, cumulative top-k."""
    # economy SVD; s is descending
    s = np.linalg.svd(matrix, compute_uv=False)
    s = np.asarray(s, dtype=float)
    energy = float(np.sum(s ** 2))
    frac = (s ** 2 / energy).tolist() if energy > 0 else [0.0] * len(s)
    cum = np.cumsum(s ** 2 / energy).tolist() if energy > 0 else [0.0] * len(s)
    return {
        "singular_values": s.tolist(),
        "fraction_variance": frac,
        "cumulative_variance": cum,
        f"cumulative_top_{top}": cum[: top],
        "total_energy_sumsq": energy,
    }


def main():
    ap = argparse.ArgumentParser(description="Spike A — escalation-axis union SVD")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="OpenAI-compatible embeddings base URL (default :8082 /v1)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="embedding model id (default embeddinggemma-300M-F32.gguf)")
    ap.add_argument("--grid", default=str(DEFAULT_GRID),
                    help="path to escalation_grid.yaml")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="JSON results output path")
    args = ap.parse_args()

    grid_path = Path(args.grid)
    with open(grid_path) as f:
        grid = yaml.safe_load(f)

    # --- collect pairs ---------------------------------------------------
    axis_pairs = []  # (axis, id, level, escalated)
    for axis in AXES:
        for item in grid[axis]:
            axis_pairs.append((axis, item["id"], item["level"], item["escalated"]))
    mood_pairs = []  # (id, level, escalated)
    for item in grid["mood_variants"]:
        mood_pairs.append((item["id"], item["level"], item["escalated"]))

    n_axis = len(axis_pairs)
    n_mood = len(mood_pairs)
    print(f"[grid] {n_axis} axis pairs, {n_mood} mood pairs", file=sys.stderr)

    adapter = LMStudioAdapter(model_name=args.model, base_url=args.base_url)

    # --- embed (fail loudly, no fabrication) -----------------------------
    all_texts = []
    for _, _, lvl, esc in axis_pairs:
        all_texts.extend([lvl, esc])
    for _, lvl, esc in mood_pairs:
        all_texts.extend([lvl, esc])

    n_expected = len(all_texts)
    print(f"[embed] requesting {n_expected} embeddings from {args.base_url} "
          f"model={args.model}", file=sys.stderr)
    try:
        raw = adapter.embed_batch(all_texts)
    except Exception as e:
        print(f"[FATAL] embedding request failed: {type(e).__name__}: {e}",
              file=sys.stderr)
        sys.exit(2)

    raw = np.asarray(raw, dtype=float)
    n_success = raw.shape[0]
    if n_success != n_expected:
        print(f"[FATAL] expected {n_expected} embeddings, got {n_success}",
              file=sys.stderr)
        sys.exit(3)

    dim = raw.shape[1]
    # diagnostics: degenerate / NaN vectors BEFORE normalization
    pre_norms = np.linalg.norm(raw, axis=1)
    n_nan = int(np.isnan(raw).any(axis=1).sum())
    n_zero = int((pre_norms == 0).sum())
    print(f"[embed] dim={dim} success={n_success}/{n_expected} "
          f"nan_rows={n_nan} zero_rows={n_zero}", file=sys.stderr)

    # unit-normalize all
    vecs = np.array([l2_normalize(v) for v in raw])
    post_norms = np.linalg.norm(vecs, axis=1)
    # how many are unit norm (within tol)
    unit_ok = int(np.sum(np.isclose(post_norms, 1.0, atol=1e-6)))

    # --- split back out --------------------------------------------------
    axis_vecs = vecs[: 2 * n_axis].reshape(n_axis, 2, dim)   # [pair, (level,esc), dim]
    mood_vecs = vecs[2 * n_axis:].reshape(n_mood, 2, dim)

    # --- deltas ----------------------------------------------------------
    axis_deltas = axis_vecs[:, 1, :] - axis_vecs[:, 0, :]    # (24, dim)
    mood_deltas = mood_vecs[:, 1, :] - mood_vecs[:, 0, :]    # (4, dim)
    axis_labels = [p[0] for p in axis_pairs]
    axis_ids = [p[1] for p in axis_pairs]
    delta_mags = np.linalg.norm(axis_deltas, axis=1)         # ||d_i||

    # --- 3. union SVD raw + centered -------------------------------------
    svd_raw = svd_spectrum(axis_deltas, top=5)
    centered = axis_deltas - axis_deltas.mean(axis=0, keepdims=True)
    svd_centered = svd_spectrum(centered, top=5)

    # --- 4. per-axis mean deltas + pairwise cosines ----------------------
    axis_means = {}
    axis_means_norm = {}
    for axis in AXES:
        idx = [i for i, lab in enumerate(axis_labels) if lab == axis]
        m = axis_deltas[idx].mean(axis=0)
        axis_means[axis] = m
        axis_means_norm[axis] = l2_normalize(m)

    inter_axis_cos = {}
    for a, b in itertools.combinations(AXES, 2):
        inter_axis_cos[f"{a}.{b}"] = float(
            np.dot(axis_means_norm[a], axis_means_norm[b])
        )

    # --- 5. within-axis coherence ----------------------------------------
    within_axis_coh = {}
    for axis in AXES:
        idx = [i for i, lab in enumerate(axis_labels) if lab == axis]
        normed = np.array([l2_normalize(axis_deltas[i]) for i in idx])
        cosines = []
        for i, j in itertools.combinations(range(len(idx)), 2):
            cosines.append(float(np.dot(normed[i], normed[j])))
        within_axis_coh[axis] = float(np.mean(cosines))

    # --- 6. length confound ----------------------------------------------
    char_diffs = []
    for _, _, lvl, esc in axis_pairs:
        char_diffs.append(len(esc) - len(lvl))
    char_diffs = np.array(char_diffs)

    token_diffs = None
    tokenize_error = None
    try:
        tdiffs = []
        for _, _, lvl, esc in axis_pairs:
            tdiffs.append(adapter.count_tokens(esc) - adapter.count_tokens(lvl))
        token_diffs = np.array(tdiffs)
    except Exception as e:
        tokenize_error = f"{type(e).__name__}: {e}"
        print(f"[tokenize] unavailable: {tokenize_error}", file=sys.stderr)

    r_char = pearson(char_diffs, delta_mags)
    r_token = pearson(token_diffs, delta_mags) if token_diffs is not None else None

    # --- 7. mood projection (held-out) -----------------------------------
    mood_proj = {}
    for k, (mid, _, _) in enumerate(mood_pairs):
        md = l2_normalize(mood_deltas[k])
        row = {axis: float(np.dot(md, axis_means_norm[axis])) for axis in AXES}
        row["argmax_axis"] = max(AXES, key=lambda a: row[a])
        mood_proj[mid] = row

    # --- assemble results ------------------------------------------------
    results = {
        "config": {
            "base_url": args.base_url,
            "model": args.model,
            "grid": str(grid_path),
            "dim": dim,
        },
        "embedding_health": {
            "n_expected": n_expected,
            "n_success": n_success,
            "n_failed": n_expected - n_success,
            "nan_rows": n_nan,
            "zero_rows": n_zero,
            "unit_norm_count": unit_ok,
            "unit_norm_all": unit_ok == n_success,
            "pre_norm_min": float(pre_norms.min()),
            "pre_norm_max": float(pre_norms.max()),
            "pre_norm_mean": float(pre_norms.mean()),
        },
        "n_axis_pairs": n_axis,
        "n_mood_pairs": n_mood,
        "axis_ids": axis_ids,
        "axis_labels": axis_labels,
        "delta_magnitudes": delta_mags.tolist(),
        "svd_raw": svd_raw,
        "svd_centered": svd_centered,
        "inter_axis_mean_cosines": inter_axis_cos,
        "within_axis_coherence": within_axis_coh,
        "length_confound": {
            "char_diffs": char_diffs.tolist(),
            "token_diffs": token_diffs.tolist() if token_diffs is not None else None,
            "tokenize_error": tokenize_error,
            "pearson_r_char_vs_mag": r_char,
            "pearson_r_token_vs_mag": r_token,
        },
        "mood_projection": mood_proj,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[done] results -> {out_path}", file=sys.stderr)

    # compact stdout summary
    def fmt(xs, n=6):
        return ", ".join(f"{v:.4f}" for v in xs[:n])

    print("\n=== SPIKE A SUMMARY ===")
    print(f"embeddings: {n_success}/{n_expected} ok, unit-norm: "
          f"{unit_ok}/{n_success}, nan={n_nan} zero={n_zero}")
    print(f"SVD RAW frac-var (top6): {fmt(svd_raw['fraction_variance'])}")
    print(f"SVD RAW cum top5: {fmt(svd_raw['cumulative_top_5'], 5)}")
    print(f"SVD CENTERED frac-var (top6): {fmt(svd_centered['fraction_variance'])}")
    print(f"SVD CENTERED cum top5: {fmt(svd_centered['cumulative_top_5'], 5)}")
    print(f"inter-axis-mean cosines: {inter_axis_cos}")
    print(f"within-axis coherence: {within_axis_coh}")
    print(f"length confound r(char)={r_char} r(token)={r_token}")
    print("mood projection (cos vs tone/urg/imp):")
    for mid, row in mood_proj.items():
        print(f"  {mid:18s} tone={row['tone']:+.3f} urgency={row['urgency']:+.3f} "
              f"importance={row['importance']:+.3f}  -> {row['argmax_axis']}")


if __name__ == "__main__":
    main()
