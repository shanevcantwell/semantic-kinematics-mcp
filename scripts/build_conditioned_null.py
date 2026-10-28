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
  1. Stream verified canonical vault rows through the TVI query CLI, scoped to
     ``input_key=base`` by default. Reservoir-subsample turns, fixed seed.
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
from semantic_kinematics.bearing.conditioned import (
    WindowTooLong,
    conditioned_vectors,
    length_bucket,
)
from semantic_kinematics.bearing.phrase_segment import segment
from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter

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
CONSUMER_NAME = "semantic-kinematics.build_conditioned_null"
CONSUMER_VERSION = "1.0"
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


def _percentiles(mags: np.ndarray) -> dict:
    return {
        "p50": float(np.percentile(mags, 50)),
        "p90": float(np.percentile(mags, 90)),
        "p95": float(np.percentile(mags, 95)),
        "p99": float(np.percentile(mags, 99)),
    }


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

    try:
        input_keys = _normalize_input_keys(args.input_keys or DEFAULT_INPUT_KEYS)
    except ValueError as exc:
        ap.error(str(exc))

    print(
        f"Reservoir-sampling {args.turns} turns (seed={args.seed}) "
        f"from canonical DB {args.corpus_db} input_keys={list(input_keys)} ..."
    )
    try:
        turns, total_usable, source_provenance = _sample_canonical_turns(
            args.corpus_db,
            canonical_root=args.canonical_root,
            input_keys=input_keys,
            k=args.turns,
            seed=args.seed,
            executable=args.tvi_executable,
        )
    except CanonicalInputError as exc:
        print(f"FATAL: canonical corpus query failed: {exc}", file=sys.stderr)
        return 2
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

    produced_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
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
            "n_turns_sampled": len(turns),
            "n_turns_contributing": int(n_turns_contributing),
            "n_turns_skipped_toolong": int(n_turns_skipped_toolong),
            "n_deltas": int(n_deltas_total),
            "source_path": args.corpus_db,
            "corpus_provenance": _corpus_provenance(
                source_provenance,
                input_keys=input_keys,
                produced_at=produced_at,
            ),
            "base_url": EMBED_BASE_URL,
            "sampling": (
                f"reservoir seed={args.seed} turns={args.turns} "
                f"min_turn_chars={MIN_TURN_CHARS}"
            ),
            "date": produced_at,
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
