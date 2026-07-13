#!/usr/bin/env python3
"""
Measure the "cone" — anisotropy diagnostics for the nv4096 corpus embeddings.

REPORTS NUMBERS ONLY. No go/no-go verdict, no prose interpretation — that is
the caller's call.

Steps:
  1. Stream the corpus JSONL line by line, json-parse each. Build a dict
     chunk_id -> embedding with dedup-keep-last (a later line for the same
     chunk_id overwrites an earlier one), matching summarize_corpus.py /
     embed_status.py conventions.
  2. Drop any all-zero embedding (every component == 0.0) from the survivors —
     these are `_failed`/retry-superseded markers. Count how many dropped.
  3. Stack survivors into a float32 array X of shape (N, 4096). Report N.
  4. Verify unit-normalization: report min/mean/max of the per-row L2 norm.
  5. Compute (all linear algebra in float64):
     - mean vector mu = X.mean(axis=0); ||mu|| and ||mu||^2
     - mean pairwise cosine over a random sample of rows (sanity check vs ||mu||^2)
     - isotropic baselines 1/sqrt(N) and 1/sqrt(4096)
     - eigenspectrum of the UNCENTERED second-moment matrix S = (X^T X)/N
       and the CENTERED covariance matrix C = (Xc^T Xc)/N, both via
       np.linalg.eigh on the 4096x4096 matrix (never a full SVD of X).
     - for each spectrum: participation ratio, total variance (trace),
       top-1 fraction, cumulative variance fraction at k in {1,5,10,34,50,100},
       first 10 raw eigenvalues.

Usage:
    python scripts/measure_cone.py
    python scripts/measure_cone.py --corpus <path> --pairwise-sample 2000 --seed 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS = (
    "../thought-vault-integration/output/thought-vault-integration-data/"
    "nv4096/corpus_4096.jsonl"
)

EMBED_DIM = 4096
CUM_K = (1, 5, 10, 34, 50, 100)


def load_corpus(corpus_path: Path):
    """Stream-parse the JSONL corpus with dedup-keep-last, dropping all-zero rows.

    Returns (chunk_ids, X_float32, counts_dict).
    """
    raw_lines = 0
    parse_errors = 0
    by_id = {}  # chunk_id -> embedding (list[float]) ; later line wins

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_lines += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1
                continue
            cid = obj.get("chunk_id")
            emb = obj.get("embedding")
            if cid is None or emb is None:
                continue
            by_id[cid] = emb  # dedup-keep-last

    distinct_ids = len(by_id)

    ids = []
    rows = []
    zero_dropped = 0
    for cid, emb in by_id.items():
        arr = np.asarray(emb, dtype=np.float32)
        if arr.shape[0] != EMBED_DIM:
            # Not part of the spec's drop criteria, but cannot be stacked;
            # treat as a zero/malformed drop and count separately below.
            zero_dropped += 1
            continue
        if not np.any(arr):
            zero_dropped += 1
            continue
        ids.append(cid)
        rows.append(arr)

    X = np.stack(rows, axis=0).astype(np.float32) if rows else np.zeros(
        (0, EMBED_DIM), dtype=np.float32
    )

    counts = {
        "raw_lines": raw_lines,
        "parse_errors": parse_errors,
        "distinct_chunk_ids": distinct_ids,
        "zero_or_malformed_dropped": zero_dropped,
        "n_used": X.shape[0],
    }
    return ids, X, counts


def eig_spectrum_report(lam_desc: np.ndarray, label: str):
    """Given descending eigenvalues (float64), compute the report block."""
    lam = np.clip(lam_desc, 0.0, None)  # guard tiny negative numerical noise
    total = float(np.sum(lam))
    sumsq = float(np.sum(lam ** 2))
    pr = (total ** 2) / sumsq if sumsq > 0 else float("nan")
    top1_frac = float(lam[0] / total) if total > 0 else float("nan")

    cum = np.cumsum(lam)
    cum_frac = {}
    for k in CUM_K:
        k_eff = min(k, lam.shape[0])
        cum_frac[k] = float(cum[k_eff - 1] / total) if total > 0 else float("nan")

    return {
        "label": label,
        "participation_ratio": pr,
        "total_variance_trace": total,
        "top1_eigenvalue_fraction": top1_frac,
        "cumulative_variance_fraction": cum_frac,
        "first_10_eigenvalues": lam_desc[:10].tolist(),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Measure isotropy/anisotropy ('cone') diagnostics for the "
        "nv4096 corpus embeddings. REPORTS NUMBERS ONLY."
    )
    ap.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="path to corpus JSONL, resolved relative to repo root if not absolute "
        f"(default: {DEFAULT_CORPUS})",
    )
    ap.add_argument(
        "--pairwise-sample",
        type=int,
        default=2000,
        help="sample size for the mean-pairwise-cosine estimate (default 2000)",
    )
    ap.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = (REPO_ROOT / corpus_path).resolve()

    print(f"[load] corpus = {corpus_path}", file=sys.stderr)
    if not corpus_path.exists():
        print(f"[FATAL] corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(2)

    ids, X, counts = load_corpus(corpus_path)
    N = X.shape[0]
    print(f"[load] N used = {N}", file=sys.stderr)

    if N == 0:
        print("[FATAL] no usable rows after load/drop", file=sys.stderr)
        sys.exit(3)

    # --- B. norm check -----------------------------------------------------
    row_norms = np.linalg.norm(X.astype(np.float64), axis=1)
    norm_min = float(row_norms.min())
    norm_mean = float(row_norms.mean())
    norm_max = float(row_norms.max())

    # --- C. mean vector ------------------------------------------------------
    Xf64_for_mean = X.astype(np.float64)
    mu = Xf64_for_mean.mean(axis=0)
    mu_norm = float(np.linalg.norm(mu))
    mu_norm_sq = mu_norm ** 2

    # --- D. mean pairwise cosine (random sample, distinct pairs) ------------
    rng = np.random.default_rng(args.seed)
    sample_size = min(args.pairwise_sample, N)
    sample_idx = rng.choice(N, size=sample_size, replace=False)
    Xs = X[sample_idx].astype(np.float64)
    # Since rows are (approximately) unit-normalized, dot product == cosine.
    gram = Xs @ Xs.T
    iu = np.triu_indices(sample_size, k=1)
    pairwise_cosines = gram[iu]
    mean_pairwise_cosine = float(pairwise_cosines.mean())
    n_pairs = pairwise_cosines.shape[0]

    # --- E. isotropic baselines ----------------------------------------------
    iso_sqrt_n = 1.0 / np.sqrt(N)
    iso_sqrt_dim = 1.0 / np.sqrt(EMBED_DIM)

    # --- F. eigenspectra via 4096x4096 second-moment / covariance -----------
    Xf64 = X.astype(np.float64)

    # Uncentered second moment: S = (X^T X) / N
    S = (Xf64.T @ Xf64) / N
    lam_uncentered = np.linalg.eigh(S)[0][::-1]  # ascending -> descending

    # Centered covariance: C = (Xc^T Xc) / N
    Xc = Xf64 - mu
    C = (Xc.T @ Xc) / N
    lam_centered = np.linalg.eigh(C)[0][::-1]

    report_uncentered = eig_spectrum_report(lam_uncentered, "UNCENTERED (second moment about origin, S = X^T X / N)")
    report_centered = eig_spectrum_report(lam_centered, "CENTERED (covariance, mean removed, C = Xc^T Xc / N)")

    # ==========================================================================
    # PRINT REPORT
    # ==========================================================================
    print("\n=== MEASURE CONE — nv4096 corpus ===\n")

    print("-- A. Counts --")
    print(f"raw_lines            = {counts['raw_lines']}")
    print(f"parse_errors         = {counts['parse_errors']}")
    print(f"distinct_chunk_ids   = {counts['distinct_chunk_ids']}")
    print(f"zero_or_malformed_dropped = {counts['zero_or_malformed_dropped']}")
    print(f"N used               = {counts['n_used']}")

    print("\n-- B. Norm check (per-row L2 norm) --")
    print(f"min  = {norm_min:.10f}")
    print(f"mean = {norm_mean:.10f}")
    print(f"max  = {norm_max:.10f}")

    print("\n-- C. Mean vector --")
    print(f"||mu||    = {mu_norm:.10f}")
    print(f"||mu||^2  = {mu_norm_sq:.10f}")

    print("\n-- D. Mean pairwise cosine --")
    print(f"sample_size = {sample_size} (seed={args.seed}), n_distinct_pairs = {n_pairs}")
    print(f"mean_pairwise_cosine = {mean_pairwise_cosine:.10f}")
    print(f"(sanity ref ||mu||^2 = {mu_norm_sq:.10f})")

    print("\n-- E. Isotropic baselines --")
    print(f"1/sqrt(N)    = {iso_sqrt_n:.10f}   (N={N})")
    print(f"1/sqrt(4096) = {iso_sqrt_dim:.10f}")

    for rep in (report_uncentered, report_centered):
        print(f"\n-- F. Eigenspectrum: {rep['label']} --")
        print(f"participation_ratio (PR)       = {rep['participation_ratio']:.6f}")
        print(f"total_variance (trace, sum lam) = {rep['total_variance_trace']:.10f}")
        print(f"top1_eigenvalue_fraction        = {rep['top1_eigenvalue_fraction']:.6f}")
        print("cumulative_variance_fraction:")
        for k in CUM_K:
            print(f"  k={k:<4d} -> {rep['cumulative_variance_fraction'][k]:.6f}")
        print("first_10_eigenvalues (raw):")
        for i, lam in enumerate(rep["first_10_eigenvalues"]):
            print(f"  lam[{i}] = {lam:.10f}")

    print("\n-- G. Interpretation-free note --")
    print(
        "participation_ratio and cumulative_variance_fraction reported above under "
        "'UNCENTERED' correspond to the second-moment matrix S = (X^T X)/N, i.e. "
        "structure about the ORIGIN (includes the mean-direction/cone component)."
    )
    print(
        "participation_ratio and cumulative_variance_fraction reported above under "
        "'CENTERED' correspond to the covariance matrix C = (Xc^T Xc)/N with Xc = X - mu, "
        "i.e. structure about the CORPUS MEAN (mean/cone direction removed)."
    )
    print(
        "The uncentered top-1 eigenvalue/fraction is dominated by the shared mean "
        "direction (the 'cone'); the centered spectrum reports variance after that "
        "direction is subtracted out."
    )

    print("\n=== END REPORT ===")


if __name__ == "__main__":
    main()
