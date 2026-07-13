#!/usr/bin/env python3
"""
Probe the poles of the top centered principal axes of the nv4096 corpus.

REPORTS NUMBERS ONLY. No go/no-go verdict, no prose interpretation — that is
the caller's call.

Steps:
  1. Load the corpus JSONL with the same dedup-keep-last / zero-row-drop rules
     as measure_cone.py: build chunk_id -> embedding with later lines winning
     on duplicate chunk_id, then drop any all-zero (or wrong-shape) embedding
     row. Stack survivors into a float32 array X of shape (N, 4096), keeping
     an ordered list of chunk_ids parallel to X.
  2. Center: mu = X.mean(0); Xc = (X - mu) in float64.
     Compute the centered covariance C = (Xc^T Xc) / N via np.linalg.eigh on
     the 4096x4096 matrix (never a full SVD of X). Take the top 2 eigenpairs
     (v0, v1) — eigh returns ascending order, so the last two columns are the
     largest, and the very last is the largest of all.
  3. Project Xc onto v0 and v1. For each axis report the projection
     distribution (min/max/mean/std) plus a sanity check that std ~= sqrt(lam).
     Rank chunk_ids by projection value; take the top/bottom N per axis
     (12/12 for v0, 8/8 for v1). For each selected chunk_id record
     (chunk_id, projection p, sigma = p / sqrt(lam)).
  4. Stream chunks.jsonl ONCE (no readlines/json.load on the whole file) to
     join selected chunk_ids to (speaker, source, conversation_name,
     timestamp, text). In the same pass also collect speaker/source counts
     for the top-200 / bottom-200 chunk_ids by v0 projection (COMPOSITION).
     Report any selected chunk_id not found in chunks.jsonl as a missing join.

Usage:
    python scripts/probe_axis_poles.py
    python scripts/probe_axis_poles.py --corpus <path> --chunks <path>
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
DEFAULT_CHUNKS = "../thought-vault-integration/output/vectors/chunks.jsonl"

EMBED_DIM = 4096

V0_POLE_N = 12
V1_POLE_N = 8
V0_COMPOSITION_N = 200

TEXT_TRUNCATE = 240


def load_corpus(corpus_path: Path):
    """Stream-parse the JSONL corpus with dedup-keep-last, dropping all-zero rows.

    Identical semantics to measure_cone.py's load_corpus.
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


def clean_text(text: str) -> str:
    """Collapse newlines to spaces and truncate to TEXT_TRUNCATE chars."""
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) > TEXT_TRUNCATE:
        return flat[:TEXT_TRUNCATE] + "..."
    return flat


def format_pole_row(entry: dict) -> str:
    speaker = entry.get("speaker", "?")
    source = entry.get("source", "?")
    # conversation_name is occasionally a raw first-message excerpt from the
    # exporter (can contain embedded newlines / be arbitrarily long) rather
    # than a short title, so it gets the same newline-collapse + truncate
    # treatment as the text field.
    conv = clean_text(entry.get("conversation_name", "?")) or "?"
    if entry.get("missing"):
        text = "[MISSING TEXT]"
    else:
        text = clean_text(entry.get("text", ""))
    sigma = entry["sigma"]
    sign = "+" if sigma >= 0 else "-"
    return f"[sigma={sign}{abs(sigma):.2f}] {speaker}/{source} | {conv} | {text}"


def main():
    ap = argparse.ArgumentParser(
        description="Probe the poles of the top centered principal axes of the "
        "nv4096 corpus embeddings, joined to source text. REPORTS NUMBERS ONLY."
    )
    ap.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help="path to embedding corpus JSONL, resolved relative to repo root if "
        f"not absolute (default: {DEFAULT_CORPUS})",
    )
    ap.add_argument(
        "--chunks",
        default=DEFAULT_CHUNKS,
        help="path to text chunks JSONL, resolved relative to repo root if not "
        f"absolute (default: {DEFAULT_CHUNKS})",
    )
    args = ap.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.is_absolute():
        corpus_path = (REPO_ROOT / corpus_path).resolve()

    chunks_path = Path(args.chunks)
    if not chunks_path.is_absolute():
        chunks_path = (REPO_ROOT / chunks_path).resolve()

    print(f"[load] corpus = {corpus_path}", file=sys.stderr)
    if not corpus_path.exists():
        print(f"[FATAL] corpus not found: {corpus_path}", file=sys.stderr)
        sys.exit(2)
    print(f"[load] chunks = {chunks_path}", file=sys.stderr)
    if not chunks_path.exists():
        print(f"[FATAL] chunks not found: {chunks_path}", file=sys.stderr)
        sys.exit(2)

    ids, X, counts = load_corpus(corpus_path)
    N = X.shape[0]
    print(f"[load] N used = {N}", file=sys.stderr)

    if N == 0:
        print("[FATAL] no usable rows after load/drop", file=sys.stderr)
        sys.exit(3)

    # --- centered eigendecomposition -----------------------------------------
    Xf64 = X.astype(np.float64)
    mu = Xf64.mean(axis=0)
    Xc = Xf64 - mu
    C = (Xc.T @ Xc) / N
    eigenvalues, eigenvectors = np.linalg.eigh(C)  # ascending order

    lam_all = eigenvalues
    total_variance = float(np.sum(np.clip(lam_all, 0.0, None)))

    lam0 = float(eigenvalues[-1])
    v0 = eigenvectors[:, -1]
    lam1 = float(eigenvalues[-2])
    v1 = eigenvectors[:, -2]

    frac0 = lam0 / total_variance if total_variance > 0 else float("nan")
    frac1 = lam1 / total_variance if total_variance > 0 else float("nan")

    # --- projections -----------------------------------------------------------
    p0 = Xc @ v0  # shape (N,)
    p1 = Xc @ v1

    def proj_stats(p):
        return {
            "min": float(p.min()),
            "max": float(p.max()),
            "mean": float(p.mean()),
            "std": float(p.std()),
        }

    p0_stats = proj_stats(p0)
    p1_stats = proj_stats(p1)
    sqrt_lam0 = float(np.sqrt(max(lam0, 0.0)))
    sqrt_lam1 = float(np.sqrt(max(lam1, 0.0)))

    # --- rank + select poles ----------------------------------------------------
    order0 = np.argsort(p0)  # ascending
    order1 = np.argsort(p1)

    def top_bottom(order, p, n):
        bottom_idx = order[:n]
        top_idx = order[::-1][:n]
        return top_idx, bottom_idx

    v0_top_idx, v0_bot_idx = top_bottom(order0, p0, V0_POLE_N)
    v1_top_idx, v1_bot_idx = top_bottom(order1, p1, V1_POLE_N)

    # For composition: top-200 / bottom-200 by v0 projection
    v0_top200_idx, v0_bot200_idx = top_bottom(order0, p0, V0_COMPOSITION_N)

    def build_selection(idx_array, p_array, lam, ids_list):
        sel = []
        for i in idx_array:
            cid = ids_list[i]
            p = float(p_array[i])
            sigma = p / np.sqrt(lam) if lam > 0 else float("nan")
            sel.append({"chunk_id": cid, "p": p, "sigma": sigma})
        return sel

    v0_top_sel = build_selection(v0_top_idx, p0, lam0, ids)
    v0_bot_sel = build_selection(v0_bot_idx, p0, lam0, ids)
    v1_top_sel = build_selection(v1_top_idx, p1, lam1, ids)
    v1_bot_sel = build_selection(v1_bot_idx, p1, lam1, ids)

    text_selected_ids = set()
    for sel in (v0_top_sel, v0_bot_sel, v1_top_sel, v1_bot_sel):
        for entry in sel:
            text_selected_ids.add(entry["chunk_id"])

    composition_ids = set(ids[i] for i in v0_top200_idx) | set(
        ids[i] for i in v0_bot200_idx
    )

    all_wanted_ids = text_selected_ids | composition_ids

    # --- single stream pass over chunks.jsonl -----------------------------------
    text_join = {}  # chunk_id -> dict(speaker, source, conversation_name, timestamp, text)
    composition_join = {}  # chunk_id -> (speaker, source)

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Cheap pre-check to avoid json.loads on every line when possible.
            if not all_wanted_ids:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            cid = obj.get("chunk_id")
            if cid is None or cid not in all_wanted_ids:
                continue
            if cid in text_selected_ids:
                text_join[cid] = {
                    "speaker": obj.get("speaker"),
                    "source": obj.get("source"),
                    "conversation_name": obj.get("conversation_name"),
                    "timestamp": obj.get("timestamp"),
                    "text": obj.get("text"),
                }
            if cid in composition_ids:
                composition_join[cid] = {
                    "speaker": obj.get("speaker"),
                    "source": obj.get("source"),
                }

    def attach_join(sel):
        out = []
        for entry in sel:
            cid = entry["chunk_id"]
            j = text_join.get(cid)
            row = dict(entry)
            if j is None:
                row["missing"] = True
            else:
                row["missing"] = False
                row.update(j)
            out.append(row)
        return out

    v0_top_rows = attach_join(v0_top_sel)
    v0_bot_rows = attach_join(v0_bot_sel)
    v1_top_rows = attach_join(v1_top_sel)
    v1_bot_rows = attach_join(v1_bot_sel)

    missing_text_ids = sorted(
        cid for cid in text_selected_ids if cid not in text_join
    )
    missing_composition_ids = sorted(
        cid for cid in composition_ids if cid not in composition_join
    )

    # --- v0 composition: speaker / source counts for top-200 vs bottom-200 ------
    def counts_for(idx_array):
        speaker_counts = {}
        source_counts = {}
        missing = []
        for i in idx_array:
            cid = ids[i]
            j = composition_join.get(cid)
            if j is None:
                missing.append(cid)
                speaker = "[MISSING]"
                source = "[MISSING]"
            else:
                speaker = j.get("speaker")
                source = j.get("source")
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
        return speaker_counts, source_counts, missing

    top200_speaker_counts, top200_source_counts, top200_missing = counts_for(
        v0_top200_idx
    )
    bot200_speaker_counts, bot200_source_counts, bot200_missing = counts_for(
        v0_bot200_idx
    )

    def sorted_counts_str(d):
        items = sorted(d.items(), key=lambda kv: (-kv[1], str(kv[0])))
        return "{" + ", ".join(f"{repr(k)}: {v}" for k, v in items) + "}"

    # ==========================================================================
    # PRINT REPORT
    # ==========================================================================
    print("\n=== PROBE AXIS POLES — nv4096 corpus ===\n")

    print("-- 1. Load counts --")
    print(f"raw_lines                 = {counts['raw_lines']}")
    print(f"parse_errors              = {counts['parse_errors']}")
    print(f"distinct_chunk_ids        = {counts['distinct_chunk_ids']}")
    print(f"zero_or_malformed_dropped = {counts['zero_or_malformed_dropped']}")
    print(f"N used                    = {counts['n_used']}")
    print(f"total_variance (trace, sum clipped lam) = {total_variance:.10f}")
    print(f"lam0 (top eigenvalue)     = {lam0:.10f}   variance_fraction = {frac0:.6f}")
    print(f"lam1 (2nd eigenvalue)     = {lam1:.10f}   variance_fraction = {frac1:.6f}")

    print("\n-- 2. v0 projection distribution --")
    print(f"min  = {p0_stats['min']:.6f}")
    print(f"max  = {p0_stats['max']:.6f}")
    print(f"mean = {p0_stats['mean']:.10f}  (expect ~0)")
    print(f"std  = {p0_stats['std']:.6f}   sqrt(lam0) = {sqrt_lam0:.6f}   (sanity check: should match)")

    print(f"\n+v0 POLE (top {V0_POLE_N}, most positive projection):")
    for row in v0_top_rows:
        print(format_pole_row(row))

    print(f"\n-v0 POLE (bottom {V0_POLE_N}, most negative projection):")
    for row in v0_bot_rows:
        print(format_pole_row(row))

    if missing_text_ids:
        print("\n[MISSING JOINS — text] chunk_ids selected for text blocks but not found in chunks.jsonl:")
        for cid in missing_text_ids:
            print(f"  {cid}")
    else:
        print("\n[MISSING JOINS — text] none")

    print(f"\n-- 3. v0 COMPOSITION (top-{V0_COMPOSITION_N} vs bottom-{V0_COMPOSITION_N} by v0 projection) --")
    print(f"speaker counts (top-{V0_COMPOSITION_N})    = {sorted_counts_str(top200_speaker_counts)}")
    print(f"source counts  (top-{V0_COMPOSITION_N})    = {sorted_counts_str(top200_source_counts)}")
    print(f"speaker counts (bottom-{V0_COMPOSITION_N}) = {sorted_counts_str(bot200_speaker_counts)}")
    print(f"source counts  (bottom-{V0_COMPOSITION_N}) = {sorted_counts_str(bot200_source_counts)}")
    if top200_missing or bot200_missing:
        print("[MISSING JOINS — composition]:")
        for cid in top200_missing:
            print(f"  (top-{V0_COMPOSITION_N}) {cid}")
        for cid in bot200_missing:
            print(f"  (bottom-{V0_COMPOSITION_N}) {cid}")
    else:
        print(f"[MISSING JOINS — composition] none")

    print("\n-- 4. v1 projection distribution --")
    print(f"min  = {p1_stats['min']:.6f}")
    print(f"max  = {p1_stats['max']:.6f}")
    print(f"mean = {p1_stats['mean']:.10f}  (expect ~0)")
    print(f"std  = {p1_stats['std']:.6f}   sqrt(lam1) = {sqrt_lam1:.6f}   (sanity check: should match)")

    print(f"\n+v1 POLE (top {V1_POLE_N}, most positive projection):")
    for row in v1_top_rows:
        print(format_pole_row(row))

    print(f"\n-v1 POLE (bottom {V1_POLE_N}, most negative projection):")
    for row in v1_bot_rows:
        print(format_pole_row(row))

    print("\n=== END REPORT ===")


if __name__ == "__main__":
    main()
