#!/usr/bin/env python
"""The look (ADR-SKMCP-0003 Phase 4): conditioned deadpan ramp on a specimen.

Reads a PRE-REGISTERED verdict spec, builds the context-conditioned trajectory at
each ``k``, scores each step's displacement magnitude against the measured
per-(k×length×demarcator) null, and applies the pre-registered decision rule.

Verdict discipline (ADR-SKMCP-0003):
- The verdict is the **null-calibrated flag pattern**, NOT the raw deadpan/heller
  curves (those drift monotonically with ``k`` as a geometric artifact and are
  reported diagnostic-only).
- POSITIVE requires a punchline to clear ``sigma`` in its stratum and stay cleared
  **contiguously** across the usable-``k`` band, while equal-format controls and
  the punchline's immediate neighbors do NOT clear (local deadpan isolation).
- A null result means "magnitude can't separate comedic from scene discontinuity
  at equal formatting," NOT "no signal." A spike is a NEW finding, never
  vindication of the falsified NV-Embed/4096-d gist.

Usage: python scripts/ramp_deadpan.py [--base-url URL] [--n-min N] [--out PATH]
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter  # noqa: E402
from semantic_kinematics.bearing.phrase_segment import segment  # noqa: E402
from semantic_kinematics.bearing.conditioned import (  # noqa: E402
    conditioned_vectors,
    length_bucket,
)
from semantic_kinematics.bearing.jolt import load_conditioned_null  # noqa: E402
from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer  # noqa: E402
from semantic_kinematics.mcp.state_manager import StateManager  # noqa: E402

SPECIMEN = os.path.join(_REPO, "data", "absurdism", "bypass_dialogue.txt")
REGISTRATION = os.path.join(_REPO, "data", "registrations", "bypass_dialogue_registration.json")
NULL = os.path.join(_REPO, "data", "nulls", "conditioned_phrase_displacement_embeddinggemma768.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8083/v1")
    ap.add_argument("--n-min", type=int, default=200)
    ap.add_argument("--out", default=os.path.join(_REPO, "output", "ramp_deadpan_bypass_result.json"))
    args = ap.parse_args()

    reg = json.load(open(REGISTRATION))
    null = load_conditioned_null(NULL)
    phrases = segment(open(SPECIMEN).read())
    adapter = LMStudioAdapter(
        model_name="embeddinggemma-300M-F32-nonpooled", base_url=args.base_url
    )
    analyzer = TrajectoryAnalyzer(StateManager())

    sigma = reg["flag_sigma"]
    k_range = reg["k_range"]
    punch = reg["punchline_indices"]
    ctrl_eq = reg["control_equal_format_indices"]
    ctrl_ss = reg["control_scene_shift_indices"]

    # step index -> {k: {z, magnitude, backoff_level, stratum_n, cleared}}
    per_k = {}
    diag = {}  # k -> {deadpan, heller}
    for k in k_range:
        M, steps = conditioned_vectors(phrases, k, adapter)
        deltas = np.linalg.norm(np.diff(M, axis=0), axis=1)  # delta[i] lands on step i+1
        rows = {}
        for i, mag in enumerate(deltas):
            j = i + 1
            s = steps[j]
            try:
                sc = null.score_step(
                    float(mag), s.actual_k, length_bucket(s.span_tokens),
                    s.demarcator_class, n_min=args.n_min,
                )
                rows[j] = {
                    "z": round(sc.z, 3), "magnitude": round(float(mag), 4),
                    "backoff_level": sc.backoff_level, "stratum_key": sc.stratum_key,
                    "stratum_n": sc.n, "cleared": sc.z >= sigma,
                    "demarcator": s.demarcator_class, "length_bucket": length_bucket(s.span_tokens),
                }
            except ValueError as exc:
                rows[j] = {"z": None, "magnitude": round(float(mag), 4),
                           "error": str(exc).split(";")[0], "cleared": False,
                           "demarcator": s.demarcator_class}
        per_k[k] = rows
        m = analyzer.analyze_embeddings(M, labels=[s.label for s in steps])
        diag[k] = {"deadpan": round(m.deadpan_score, 3), "heller": round(m.heller_score, 3)}

    def z_at(idx, k):
        r = per_k[k].get(idx)
        return r["z"] if r and r["z"] is not None else None

    def cleared(idx, k):
        r = per_k[k].get(idx)
        return bool(r and r.get("cleared"))

    # --- apply the pre-registered decision rule -------------------------------
    kmax = max(k_range)
    findings = {}
    verdict = "NEGATIVE"
    for p in punch:
        # contiguous top-band: smallest k* (>=1) s.t. punchline cleared for all k in [k*, kmax]
        kstar = None
        for cand in sorted(x for x in k_range if x >= 1):
            band = [k for k in k_range if k >= cand]
            if band and all(cleared(p, k) for k in band):
                kstar = cand
                break
        qualifies = False
        band_controls_ok = None
        if kstar is not None:
            band = [k for k in k_range if k >= kstar]
            neighbors = [p - 1, p + 1]
            controls_clear = any(cleared(c, k) for c in ctrl_eq for k in band)
            neighbors_clear = any(cleared(nb, k) for nb in neighbors for k in band)
            band_controls_ok = {"equal_format_controls_silent": not controls_clear,
                                "neighbors_silent": not neighbors_clear}
            qualifies = (not controls_clear) and (not neighbors_clear)
        findings[p] = {
            "label": phrases[p].content.strip()[:60],
            "z_by_k": {k: z_at(p, k) for k in k_range},
            "cleared_by_k": {k: cleared(p, k) for k in k_range},
            "contiguous_kstar": kstar,
            "qualifies": qualifies,
            "band_checks": band_controls_ok,
            "verdict_cell_backoff": {k: (per_k[k].get(p) or {}).get("backoff_level") for k in k_range},
        }
        if qualifies:
            verdict = "POSITIVE"

    result = {
        "specimen": reg["specimen"],
        "sigma": sigma,
        "null": os.path.basename(NULL),
        "null_turns": load_conditioned_null(NULL).header.get("n_turns_contributing"),
        "verdict": verdict,
        "verdict_language": (
            "POSITIVE = correctly-atomed instrument detects the deadpan SHAPE (control-distinct, "
            "locally-isolated, contiguous over k); a NEW finding, not vindication of the falsified "
            "NV-Embed/4096-d gist. NEGATIVE = magnitude can't separate comedic from scene "
            "discontinuity AT EQUAL FORMATTING (not 'no signal')."
        ),
        "punchline_findings": findings,
        "equal_format_controls": {c: {k: z_at(c, k) for k in k_range} for c in ctrl_eq},
        "scene_shift_controls": {c: {k: z_at(c, k) for k in k_range} for c in ctrl_ss},
        "diagnostic_raw_curves_DO_NOT_USE_AS_VERDICT": diag,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)

    # --- console summary ------------------------------------------------------
    print(f"\nVERDICT: {verdict}   (sigma={sigma}, null={result['null_turns']} turns)")
    print("\nPunchlines (z by k; * = cleared):")
    for p, f in findings.items():
        cells = " ".join(
            f"k{k}={'-' if f['z_by_k'][k] is None else f['z_by_k'][k]}{'*' if f['cleared_by_k'][k] else ''}"
            for k in k_range
        )
        print(f"  [{p}] {f['label']!r}")
        print(f"      {cells}  | kstar={f['contiguous_kstar']} qualifies={f['qualifies']} {f['band_checks']}")
    print("\nEqual-format controls (max z over k):")
    for c in ctrl_eq:
        zs = [z_at(c, k) for k in k_range if z_at(c, k) is not None]
        print(f"  [{c}] {phrases[c].content.strip()[:40]!r}  max_z={max(zs) if zs else None}")
    print("\nScene-shift controls (max z over k):")
    for c in ctrl_ss:
        zs = [z_at(c, k) for k in k_range if z_at(c, k) is not None]
        print(f"  [{c}] {phrases[c].content.strip()[:40]!r}  max_z={max(zs) if zs else None}")
    print("\nDIAGNOSTIC ONLY (k-artifact; NOT the verdict) raw deadpan/heller:")
    for k in k_range:
        print(f"  k{k}: deadpan={diag[k]['deadpan']} heller={diag[k]['heller']}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
