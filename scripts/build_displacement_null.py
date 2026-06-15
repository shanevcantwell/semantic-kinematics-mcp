#!/usr/bin/env python
"""Build the MEASURED sentence-displacement null for the bearing-magnitude regime.

The null is the empirical distribution of WITHIN-TURN consecutive-sentence
displacement magnitudes ``||v[i+1] - v[i]||`` over real vault text, embedded on
the SAME embedder and at the SAME atom (sentence) as the test specimen. This is
the atom-matching discipline: turn-to-turn deltas are a different (larger) regime
and must NOT be used to calibrate a sentence-atom detector.

Pipeline:
  1. Stream vault chunks.jsonl (read-only). Subsample turns with a fixed seed.
  2. For each sampled turn: spaCy sentence-split (same path as TrajectoryAnalyzer).
  3. Embed consecutive sentences on :8082 (768-d, unit-norm).
  4. Collect within-turn ||delta|| (never crossing a turn boundary).
  5. Persist a self-describing JSON null to data/nulls/.

No silent defaults; no in-engine retry. A transient embed failure raises.

Usage:
    python scripts/build_displacement_null.py \
        --target-sentences 4000 --seed 1729
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import random
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer  # noqa: E402
from semantic_kinematics.mcp.state_manager import StateManager  # noqa: E402

DEFAULT_SOURCE = (
    "/home/shane/github/shanevcantwell/thought-vault-integration/"
    "output/vectors/chunks.jsonl"
)
EMBED_BACKEND = "lmstudio"
EMBED_BASE_URL = "http://localhost:8082/v1"
EMBED_MODEL = "embeddinggemma-300M-F32"
EXPECTED_DIM = 768

# Minimum chars for a turn to be worth sentence-splitting (skip terse one-liners
# like "ok" that yield no within-turn deltas anyway).
MIN_TURN_CHARS = 40


def _build_analyzer() -> TrajectoryAnalyzer:
    manager = StateManager()
    manager.set_backend(EMBED_BACKEND, base_url=EMBED_BASE_URL, model_name=EMBED_MODEL)
    return TrajectoryAnalyzer(manager)


def _reservoir_sample_turns(path: str, k: int, seed: int):
    """Reservoir-sample up to k usable turns from a streamed jsonl (read-only).

    A 'usable' turn has >= MIN_TURN_CHARS of text. Reservoir sampling gives a
    uniform sample over the stream in a single pass without loading the file.
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE, help="vault chunks.jsonl")
    ap.add_argument(
        "--target-sentences",
        type=int,
        default=4000,
        help="approximate target sentence count (3000-6000 recommended)",
    )
    ap.add_argument("--seed", type=int, default=1729)
    ap.add_argument(
        "--out",
        default=os.path.join(
            _REPO_ROOT, "data", "nulls", "sentence_displacement_embeddinggemma768.json"
        ),
    )
    args = ap.parse_args()

    if not os.path.isfile(args.source):
        print(f"FATAL: source not found: {args.source}", file=sys.stderr)
        return 2

    # Heuristic: avg ~2 within-turn deltas per usable turn -> sample ~target/2
    # turns, then trim. We oversample turns slightly to land near the target.
    target_turns = max(1, args.target_sentences // 2)

    print(f"Reservoir-sampling ~{target_turns} turns (seed={args.seed}) "
          f"from {args.source} ...")
    turns, total_usable = _reservoir_sample_turns(args.source, target_turns, args.seed)
    print(f"  sampled {len(turns)} turns out of {total_usable} usable in stream")

    analyzer = _build_analyzer()

    all_mags = []
    n_turns_contributing = 0
    n_sentences_total = 0
    n_turns_skipped_oversize = 0

    for ti, turn in enumerate(turns):
        sentences = analyzer.tokenize_sentences(turn)
        if len(sentences) < 2:
            continue  # no within-turn delta possible
        # Embed each sentence in its own request. The :8082 server has a hard
        # 2048-token physical batch ceiling; batching a whole turn's sentences
        # in one call can blow past it. Per-sentence keeps each request small AND
        # is the correct atom. A single sentence that still exceeds the ceiling
        # is pathological tokenization -- skip that turn honestly and count it,
        # rather than retrying or silently truncating (no in-engine retry rule).
        try:
            embs = np.vstack([analyzer.embed_sentences([s])[0] for s in sentences])
        except Exception as exc:  # noqa: BLE001 - surface as an honest skip count
            if "too large to process" in str(exc):
                n_turns_skipped_oversize += 1
                continue
            raise
        embs = np.asarray(embs, dtype=float)
        if embs.ndim != 2 or embs.shape[1] != EXPECTED_DIM:
            raise ValueError(
                f"turn {ti}: embedding shape {embs.shape} != [*, {EXPECTED_DIM}]"
            )
        # WITHIN-turn deltas only; np.diff never crosses turn boundary because we
        # embed one turn at a time.
        mags = np.linalg.norm(np.diff(embs, axis=0), axis=1)
        all_mags.extend(float(m) for m in mags)
        n_turns_contributing += 1
        n_sentences_total += len(sentences)

        if (ti + 1) % 200 == 0:
            print(f"  ...{ti + 1}/{len(turns)} turns, {len(all_mags)} deltas so far")

    if not all_mags:
        print("FATAL: no within-turn deltas collected.", file=sys.stderr)
        return 3

    mags = np.asarray(all_mags, dtype=float)
    pcts = {
        "p50": float(np.percentile(mags, 50)),
        "p90": float(np.percentile(mags, 90)),
        "p95": float(np.percentile(mags, 95)),
        "p99": float(np.percentile(mags, 99)),
    }

    artifact = {
        "header": {
            "regime": "bearing-magnitude",
            "atom": "sentence",
            "embedder": EMBED_MODEL,
            "dim": EXPECTED_DIM,
            "n_deltas": int(mags.size),
            "n_turns_sampled": int(len(turns)),
            "n_turns_contributing": int(n_turns_contributing),
            "n_turns_skipped_oversize": int(n_turns_skipped_oversize),
            "n_sentences": int(n_sentences_total),
            "source_path": args.source,
            "sampling": (
                f"reservoir seed={args.seed} target_turns={target_turns} "
                f"min_turn_chars={MIN_TURN_CHARS}"
            ),
            "base_url": EMBED_BASE_URL,
            "date": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        },
        "stats": {
            "mean": float(mags.mean()),
            "std": float(mags.std(ddof=0)),
            "min": float(mags.min()),
            "max": float(mags.max()),
            "percentiles": pcts,
        },
        # Full measured distribution -> exact empirical percentile-rank at score
        # time. Unit-norm vectors keep magnitudes in [0, 2]; this is small JSON.
        "magnitudes": [round(float(m), 6) for m in mags.tolist()],
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh)

    print("\n=== NULL BUILT ===")
    print(f"  out         : {args.out}")
    print(f"  n_deltas    : {mags.size}")
    print(f"  n_turns     : {n_turns_contributing} contributing "
          f"/ {len(turns)} sampled")
    print(f"  n_sentences : {n_sentences_total}")
    print(f"  mean / std  : {mags.mean():.6f} / {mags.std(ddof=0):.6f}")
    print(f"  p50/90/95/99: {pcts['p50']:.4f} / {pcts['p90']:.4f} / "
          f"{pcts['p95']:.4f} / {pcts['p99']:.4f}")
    print(f"  range       : [{mags.min():.4f}, {mags.max():.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
