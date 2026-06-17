#!/usr/bin/env python
"""Build the MEASURED context-conditioned phrase-displacement null (ADR-SKMCP-0003).

The null is the empirical distribution of WITHIN-TURN consecutive conditioned-
phrase displacement magnitudes ``||cv[i+1] - cv[i]||`` over real vault text,
embedded on the SAME embedder (embeddinggemma in ``--pooling none``) at the SAME
atom (context-conditioned phrase span) as the test specimen. Each delta is
stratified by the LANDING step's ``(actual_k, length_bucket, demarcator_class)``
because pooling-variance, context-overlap (k), and demarcator type are confounded
with the comedic signal and must be absorbed by the null before any number is read
(ADR Decision 5).

Pipeline:
  1. Stream vault chunks.jsonl (read-only). Reservoir-subsample turns, fixed seed.
  2. For each sampled turn: ``segment(turn_text)`` -> phrase list.
  3. For each k in K_RANGE: ``conditioned_vectors(phrases, k, adapter)`` -> the
     conditioned trajectory + per-phrase ConditionedStep keys.
  4. Collect within-turn ||cv[i+1]-cv[i]|| (never crossing a turn boundary),
     attributing each delta to the LANDING step (i+1)'s stratum.
  5. Persist a self-describing, per-stratum JSON null to data/nulls/.

No silent defaults; no in-engine retry. A transient embed failure raises.

Construction note: this script does NOT reimplement the conditioned-vector
construction -- it calls ``conditioned_vectors`` / ``segment`` directly, the SAME
calls the look uses, so the per-k smoothing cancels in numerator and denominator
(ADR confound #1 / read-discipline guard).

Usage:
    python scripts/build_conditioned_null.py --turns 50 --out /tmp/cond_null_smoke.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_kinematics.bearing.conditioned import (  # noqa: E402
    WindowTooLong,
    conditioned_vectors,
    length_bucket,
)
from semantic_kinematics.bearing.phrase_segment import segment  # noqa: E402
from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter  # noqa: E402

DEFAULT_SOURCE = (
    "/home/shane/github/shanevcantwell/thought-vault-integration/"
    "output/vectors/chunks.jsonl"
)
EMBED_MODEL = "embeddinggemma-300M-F32-nonpooled"
EMBED_BASE_URL = "http://localhost:8083/v1"
EXPECTED_DIM = 768
K_RANGE = [0, 1, 2, 3, 4, 5]

# Minimum chars for a turn to be worth segmenting (terse one-liners yield no
# within-turn deltas anyway).
MIN_TURN_CHARS = 40

# Length-bucket labels (mirrors conditioned.LENGTH_BUCKETS).
LENGTH_BUCKETS_LABELS = ["1-3", "4-7", "8-15", "16+"]
DEMARCATOR_CLASSES = [
    "TERM_ISOLATED",
    "TERM_FLOW",
    "INTERNAL",
    "DASH_ELLIP",
    "BREAK_BARE",
    "SET_QUOTE",
    "SET_PAREN",
    "NONE",
]


def _reservoir_sample_turns(path: str, k: int, seed: int):
    """Reservoir-sample up to k usable turns from a streamed jsonl (read-only).

    A 'usable' turn has >= MIN_TURN_CHARS of text. Single pass, no full load.
    """
    rng = random.Random(seed)
    reservoir = []
    seen = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = (rec.get("text") or "").strip()
            if len(text) < MIN_TURN_CHARS:
                continue
            seen += 1
            if len(reservoir) < k:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < k:
                    reservoir[j] = text
    return reservoir, seen


def _percentiles(mags: np.ndarray) -> dict:
    return {
        "p50": float(np.percentile(mags, 50)),
        "p90": float(np.percentile(mags, 90)),
        "p95": float(np.percentile(mags, 95)),
        "p99": float(np.percentile(mags, 99)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="vault chunks.jsonl")
    ap.add_argument(
        "--turns",
        type=int,
        default=50,
        help="number of turns to reservoir-sample (default small for safety)",
    )
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--out",
        default=os.path.join(
            _REPO_ROOT,
            "data",
            "nulls",
            "conditioned_phrase_displacement_embeddinggemma768.json",
        ),
    )
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        print(f"FATAL: source not found: {args.source}", file=sys.stderr)
        return 2

    print(f"Reservoir-sampling {args.turns} turns (seed={args.seed}) "
          f"from {args.source} ...")
    turns, total_usable = _reservoir_sample_turns(args.source, args.turns, args.seed)
    print(f"  sampled {len(turns)} turns out of {total_usable} usable in stream")

    adapter = LMStudioAdapter(model_name=EMBED_MODEL, base_url=EMBED_BASE_URL)

    # Per-stratum magnitude lists, keyed "k{K}|{lenbucket}|{demarc}".
    strata: dict = {}
    # Bias diagnostics: per-k delta count + per-k cumulative leading-overlap
    # length (sum over steps of total len(content) of the leading phrases in the
    # landing step's window). Makes high-k long-turn bias visible.
    per_k_n: dict = {k: 0 for k in K_RANGE}
    per_k_lead_overlap: dict = {k: 0 for k in K_RANGE}

    n_turns_contributing = 0
    n_turns_skipped_toolong = 0
    n_deltas_total = 0
    t0 = time.time()

    for ti, turn in enumerate(turns):
        phrases = segment(turn)
        if len(phrases) < 2:
            continue  # no within-turn delta possible

        contributed = False
        skipped_toolong = False
        for k in K_RANGE:
            try:
                matrix, steps = conditioned_vectors(phrases, k, adapter)
            except WindowTooLong:
                # Deterministic content limit (a single phrase exceeds the token
                # ceiling); k-independent so it raises at k=0 before any append.
                skipped_toolong = True
                break
            if matrix.shape[1] != EXPECTED_DIM:
                raise ValueError(
                    f"turn {ti}: conditioned dim {matrix.shape[1]} "
                    f"!= {EXPECTED_DIM}"
                )
            # WITHIN-turn consecutive deltas; matrix is one turn so np.diff never
            # crosses a turn boundary. Delta i is cv[i+1]-cv[i]; ATTRIBUTE it to
            # the LANDING step i+1.
            deltas = np.linalg.norm(np.diff(matrix, axis=0), axis=1)
            for i, mag in enumerate(deltas):
                landing = steps[i + 1]
                key = (
                    f"k{landing.actual_k}|"
                    f"{length_bucket(landing.span_tokens)}|"
                    f"{landing.demarcator_class}"
                )
                strata.setdefault(key, []).append(float(mag))
                # Bias diagnostics keyed by the REQUESTED k (the ramp index this
                # delta belongs to), separate from landing.actual_k strata keys.
                per_k_n[k] += 1
                # Leading-overlap = total content length of the leading phrases
                # in the landing step's window (those before the target).
                lead = phrases[
                    (i + 1) - landing.actual_k : (i + 1)
                ]
                per_k_lead_overlap[k] += sum(len(p.content) for p in lead)
                n_deltas_total += 1
                contributed = True

        if skipped_toolong:
            n_turns_skipped_toolong += 1
            continue
        if contributed:
            n_turns_contributing += 1

        if (ti + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = n_deltas_total / elapsed if elapsed > 0 else 0.0
            print(f"  ...{ti + 1}/{len(turns)} turns, {n_deltas_total} deltas, "
                  f"{rate:.1f} deltas/s")

    elapsed = time.time() - t0

    if not strata:
        print("FATAL: no within-turn deltas collected.", file=sys.stderr)
        return 3

    # Aggregate coarser backoff parents so the scorer's sparsity backoff has
    # cells to find: the finest key is "k{K}|{len}|{demarc}"; the scorer coarsens
    # to "k{K}|{len}" then "k{K}". A landing step ALWAYS has a length+demarcator,
    # so only the finest keys arise naturally -- the parents are unions of their
    # children, built here so backoff is not a dead path.
    agg: dict = {}
    for key, mags_list in strata.items():
        ak, lb, _dem = key.split("|", 2)
        agg.setdefault(key, []).extend(mags_list)              # finest
        agg.setdefault(f"{ak}|{lb}", []).extend(mags_list)     # (k, length)
        agg.setdefault(ak, []).extend(mags_list)               # (k)

    # Build per-stratum summaries.
    stratum_blocks = {}
    for key, mags_list in sorted(agg.items()):
        mags = np.asarray(mags_list, dtype=float)
        stratum_blocks[key] = {
            "mean": float(mags.mean()),
            "std": float(mags.std(ddof=0)),
            "n": int(mags.size),
            "percentiles": _percentiles(mags),
            "sorted_magnitudes": [round(float(m), 6) for m in np.sort(mags).tolist()],
        }

    bias_diagnostics = {
        "per_k_n": {str(k): per_k_n[k] for k in K_RANGE},
        "per_k_mean_leading_overlap_len": {
            str(k): (per_k_lead_overlap[k] / per_k_n[k] if per_k_n[k] else 0.0)
            for k in K_RANGE
        },
    }

    artifact = {
        "header": {
            "regime": "bearing-magnitude-conditioned",
            "atom": "phrase-conditioned",
            "pooling": "span-mean",
            "embedder": EMBED_MODEL,
            "dim": EXPECTED_DIM,
            "k_range": K_RANGE,
            "length_buckets": LENGTH_BUCKETS_LABELS,
            "demarcator_classes": DEMARCATOR_CLASSES,
            "n_turns_sampled": int(len(turns)),
            "n_turns_contributing": int(n_turns_contributing),
            "n_turns_skipped_toolong": int(n_turns_skipped_toolong),
            "n_deltas": int(n_deltas_total),
            "source_path": args.source,
            "base_url": EMBED_BASE_URL,
            "sampling": (
                f"reservoir seed={args.seed} turns={args.turns} "
                f"min_turn_chars={MIN_TURN_CHARS}"
            ),
            "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "strata": stratum_blocks,
        "bias_diagnostics": bias_diagnostics,
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh)

    rate = n_deltas_total / elapsed if elapsed > 0 else 0.0
    print("\n=== CONDITIONED NULL BUILT ===")
    print(f"  out          : {args.out}")
    print(f"  n_deltas     : {n_deltas_total}")
    print(f"  n_turns      : {n_turns_contributing} contributing "
          f"/ {len(turns)} sampled")
    print(f"  n_strata     : {len(stratum_blocks)}")
    print(f"  elapsed      : {elapsed:.1f}s ({rate:.1f} deltas/s)")
    print(f"  per-k n      : {bias_diagnostics['per_k_n']}")
    print(f"  per-k lead-ov: "
          f"{ {k: round(v, 1) for k, v in bias_diagnostics['per_k_mean_leading_overlap_len'].items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
