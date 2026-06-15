"""Bearing regime: per-step displacement-magnitude ("jolt") detection.

This is the AXIS-FREE half of Spike B (issue #31). A jolt is scored purely on
per-step displacement magnitude ``||v[i+1] - v[i]||`` against a MEASURED
real-text displacement baseline (the "null"). There is no projection axis here.
"""

from semantic_kinematics.bearing.jolt import (
    DisplacementNull,
    JoltResult,
    JoltStep,
    load_null,
    score_jolts,
)

__all__ = [
    "DisplacementNull",
    "JoltResult",
    "JoltStep",
    "load_null",
    "score_jolts",
]
