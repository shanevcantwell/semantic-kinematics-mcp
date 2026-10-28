#!/usr/bin/env python
"""Build the MEASURED sentence-displacement null for the bearing-magnitude regime.

The null is the empirical distribution of WITHIN-TURN consecutive-sentence
displacement magnitudes ``||v[i+1] - v[i]||`` over real vault text, embedded on
the SAME embedder and at the SAME atom (sentence) as the test specimen. This is
the atom-matching discipline: turn-to-turn deltas are a different (larger) regime
and must NOT be used to calibrate a sentence-atom detector.

Pipeline:
  1. Stream verified canonical vault rows through the TVI query CLI, scoped to
     ``input_key=base`` by default. Subsample turns with a fixed seed.
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
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.tvi_canonical_input import (
    DEFAULT_EXECUTABLE,
    CanonicalInputError,
    CanonicalQueryProvenance,
    open_canonical_chunks,
)
from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer
from semantic_kinematics.mcp.state_manager import StateManager

# The live thought-vault source is the promoted canonical database. The
# environment override keeps deployment-specific locations out of the CLI while
# retaining a usable default for the established /srv layout (issue #34).
DEFAULT_CORPUS_DB = os.environ.get(
    "THOUGHT_VAULT_CANONICAL_DB",
    "/srv/dev/shanevcantwell/thought-vault-integration/output/canonical/corpus.db",
)
DEFAULT_CANONICAL_ROOT = os.environ.get("THOUGHT_VAULT_CANONICAL_ROOT")
DEFAULT_TVI_EXECUTABLE = os.environ.get(
    "TVI_CANONICAL_CORPUS_EXECUTABLE", DEFAULT_EXECUTABLE
)
DEFAULT_INPUT_KEYS = ("base",)
CONSUMER_NAME = "semantic-kinematics.build_displacement_null"
CONSUMER_VERSION = "1.0"
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


def _iter_usable_texts(rows: Iterable[Mapping[str, Any]]) -> Iterator[str]:
    """Yield the old null-builder population from rows in source order."""
    for row in rows:
        text = (row.get("text") or "").strip()
        if len(text) >= MIN_TURN_CHARS:
            yield text


def _reservoir_sample_turns(
    rows: Iterable[Mapping[str, Any]], k: int, seed: int
) -> tuple[list[str], int]:
    """Reservoir-sample usable turns in query-door order without buffering rows."""
    rng = random.Random(seed)
    reservoir: list[str] = []
    seen = 0
    for text in _iter_usable_texts(rows):
        seen += 1
        if len(reservoir) < k:
            reservoir.append(text)
        else:
            j = rng.randint(0, seen - 1)
            if j < k:
                reservoir[j] = text
    return reservoir, seen


def _normalize_input_keys(input_keys: Iterable[str]) -> tuple[str, ...]:
    keys = tuple(sorted(input_keys))
    if not keys:
        raise ValueError("at least one --input-key is required")
    if len(set(keys)) != len(keys):
        raise ValueError("--input-key values must not contain duplicates")
    return keys


def _sample_canonical_turns(
    corpus_db: str,
    *,
    canonical_root: str | None,
    input_keys: tuple[str, ...],
    k: int,
    seed: int,
    executable: str,
) -> tuple[list[str], int, CanonicalQueryProvenance]:
    """Sample one verified TVI traversal; acquisition failures propagate."""
    with open_canonical_chunks(
        corpus_db,
        canonical_root=canonical_root,
        input_keys=input_keys,
        executable=executable,
    ) as stream:
        provenance = stream.provenance
        turns, seen = _reservoir_sample_turns(stream, k, seed)
    return turns, seen, provenance


def _corpus_provenance(
    provenance: CanonicalQueryProvenance,
    *,
    input_keys: tuple[str, ...],
    produced_at: str,
) -> dict[str, Any]:
    block = provenance.as_dict()
    block.update(
        {
            "source_kind": "tvi_canonical_db",
            "normalized_input_keys": list(input_keys),
            "normalized_filters": {
                "channels": [],
                "speakers": [],
                "conversations": [],
                "nonblank_text": None,
                "time_roles": [],
            },
            "consumer": {"name": CONSUMER_NAME, "version": CONSUMER_VERSION},
            "produced_at": produced_at,
        }
    )
    return block


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--corpus-db",
        default=DEFAULT_CORPUS_DB,
        help="promoted TVI canonical corpus.db (verified-open; no JSONL fallback)",
    )
    ap.add_argument(
        "--canonical-root",
        default=DEFAULT_CANONICAL_ROOT,
        help="optional trusted TVI canonical root for direct generation paths",
    )
    ap.add_argument(
        "--input-key",
        action="append",
        dest="input_keys",
        help="canonical acquisition key (repeatable; default: base)",
    )
    ap.add_argument(
        "--tvi-executable",
        default=DEFAULT_TVI_EXECUTABLE,
        help="installed tvi-canonical-corpus executable",
    )
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

    try:
        input_keys = _normalize_input_keys(args.input_keys or DEFAULT_INPUT_KEYS)
    except ValueError as exc:
        ap.error(str(exc))

    # Heuristic: avg ~2 within-turn deltas per usable turn -> sample ~target/2
    # turns, then trim. We oversample turns slightly to land near the target.
    target_turns = max(1, args.target_sentences // 2)

    print(
        f"Reservoir-sampling ~{target_turns} turns (seed={args.seed}) "
        f"from canonical DB {args.corpus_db} input_keys={list(input_keys)} ..."
    )
    try:
        turns, total_usable, source_provenance = _sample_canonical_turns(
            args.corpus_db,
            canonical_root=args.canonical_root,
            input_keys=input_keys,
            k=target_turns,
            seed=args.seed,
            executable=args.tvi_executable,
        )
    except CanonicalInputError as exc:
        print(f"FATAL: canonical corpus query failed: {exc}", file=sys.stderr)
        return 2
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
        except Exception as exc:  # surface as an honest skip count
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

    produced_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    artifact = {
        "header": {
            "regime": "bearing-magnitude",
            "atom": "sentence",
            "embedder": EMBED_MODEL,
            "dim": EXPECTED_DIM,
            "n_deltas": int(mags.size),
            "n_turns_sampled": len(turns),
            "n_turns_contributing": int(n_turns_contributing),
            "n_turns_skipped_oversize": int(n_turns_skipped_oversize),
            "n_sentences": int(n_sentences_total),
            "source_path": args.corpus_db,
            "corpus_provenance": _corpus_provenance(
                source_provenance,
                input_keys=input_keys,
                produced_at=produced_at,
            ),
            "sampling": (
                f"reservoir seed={args.seed} target_turns={target_turns} "
                f"min_turn_chars={MIN_TURN_CHARS}"
            ),
            "base_url": EMBED_BASE_URL,
            "date": produced_at,
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
