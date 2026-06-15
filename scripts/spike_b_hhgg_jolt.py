#!/usr/bin/env python
"""Spike B (axis-free half, issue #31): jolt detection on the HHGG Prosser/Arthur
bypass argument, scored against the MEASURED sentence-displacement null.

Atom-matching: the specimen is SENTENCE-atom; it is scored against the
sentence-atom, embeddinggemma-768 null at data/nulls/. No projection axis is
involved -- a "jolt" is purely ``||v[i+1] - v[i]||`` standing sigma above a real
real-text sentence-displacement baseline.

Verdict against Spike B kill criteria:
  - jolts a human reads in the argument clear the null at meaningful sigma
    -> instrument detects its target (provisional pass).
  - nothing clears the null -> instrument does not detect its target at
    sentence-atom in embeddinggemma-768 -> LOUD FAIL (stop). Reported plainly.

Usage:
    python scripts/spike_b_hhgg_jolt.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from semantic_kinematics.bearing.jolt import load_null, score_jolts  # noqa: E402
from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer  # noqa: E402
from semantic_kinematics.mcp.state_manager import StateManager  # noqa: E402

SPECIMEN_PATH = os.path.join(_REPO_ROOT, "data", "absurdism", "bypass_dialogue.txt")
NULL_PATH = os.path.join(
    _REPO_ROOT, "data", "nulls", "sentence_displacement_embeddinggemma768.json"
)
EMBED_BACKEND = "lmstudio"
EMBED_BASE_URL = "http://localhost:8082/v1"
EMBED_MODEL = "embeddinggemma-300M-F32"
THRESHOLD_SIGMA = 3.0


def _truncate(text, n=110):
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "..."


def main() -> int:
    if not os.path.isfile(SPECIMEN_PATH):
        print(f"FATAL: specimen not found: {SPECIMEN_PATH}", file=sys.stderr)
        return 2

    # Hard-fail (no silent default) if the null artifact is absent.
    null = load_null(NULL_PATH)
    print("=== NULL ===")
    print(f"  path     : {NULL_PATH}")
    print(f"  regime   : {null.regime} | atom: {null.atom} | "
          f"embedder: {null.embedder} | dim: {null.dim}")
    print(f"  n_deltas : {null.n_deltas}")
    print(f"  mean/std : {null.mean:.6f} / {null.std:.6f}")
    print(f"  pcts     : {null.percentiles}")

    # Embed the specimen sentence-wise on the SAME embedder as the null.
    manager = StateManager()
    manager.set_backend(EMBED_BACKEND, base_url=EMBED_BASE_URL, model_name=EMBED_MODEL)
    analyzer = TrajectoryAnalyzer(manager)

    with open(SPECIMEN_PATH, "r", encoding="utf-8") as fh:
        text = fh.read()

    sentences = analyzer.tokenize_sentences(text)
    print(f"\n=== SPECIMEN: {os.path.basename(SPECIMEN_PATH)} ===")
    print(f"  sentences: {len(sentences)}")
    if len(sentences) < 2:
        print("FATAL: need >= 2 sentences.", file=sys.stderr)
        return 3

    # Per-sentence embedding to respect the server's batch-token ceiling and the
    # sentence atom (same path used to build the null).
    embs = np.vstack([analyzer.embed_sentences([s])[0] for s in sentences])

    result = score_jolts(
        embs, null, threshold_sigma=THRESHOLD_SIGMA, labels=sentences
    )

    print(f"\n=== JOLT SCORING (threshold {THRESHOLD_SIGMA} sigma) ===")
    print(f"  steps           : {result.n_steps - 1} transitions over "
          f"{result.n_steps} sentences")
    print(f"  peak sigma      : {result.peak_z:.3f} @ step {result.peak_index} "
          f"(sentence #{result.peak_index + 1})")
    peak_step = result.steps[result.peak_index]
    print(f"    peak lands on : \"{_truncate(peak_step.label)}\"")
    print(f"    peak mag/pct  : {peak_step.magnitude:.4f} / "
          f"{peak_step.percentile:.3f}%")
    print(f"  jolts >= {THRESHOLD_SIGMA}s : {len(result.flagged)}")

    if result.flagged:
        print("\n  FLAGGED JOLTS (human-readable):")
        for s in result.flagged:
            print(f"    step {s.index:>2} -> sentence #{s.index + 1} | "
                  f"sigma {s.z:6.3f} | pct {s.percentile:7.3f}% | "
                  f"mag {s.magnitude:.4f}")
            print(f'        "{_truncate(s.label)}"')

    # Show the top-5 steps regardless of threshold for context.
    top = sorted(result.steps, key=lambda s: s.z, reverse=True)[:5]
    print("\n  TOP-5 steps by sigma (context):")
    for s in top:
        print(f"    step {s.index:>2} | sigma {s.z:6.3f} | pct {s.percentile:7.3f}% "
              f"| \"{_truncate(s.label, 70)}\"")

    print("\n=== VERDICT (Spike B kill criteria) ===")
    if result.flagged:
        print(f"  PROVISIONAL PASS: {len(result.flagged)} jolt(s) clear the null "
              f"at >= {THRESHOLD_SIGMA} sigma (peak {result.peak_z:.2f}s).")
        print("  The instrument flags discrete sentence-level jolts above a real")
        print("  sentence-displacement baseline. Read the flagged sentences above")
        print("  to judge whether they are the human-legible argument beats.")
    else:
        print(f"  LOUD FAIL: NOTHING clears the null at >= {THRESHOLD_SIGMA} sigma "
              f"(peak only {result.peak_z:.2f}s).")
        print("  At sentence atom in embeddinggemma-768, per-step displacement")
        print("  magnitude does NOT separate the argument's jolts from ordinary")
        print("  real-text sentence-to-sentence motion. STOP per kill criterion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
