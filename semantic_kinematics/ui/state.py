"""
State management for the semantic-kinematics UI.

Provides:
- Singleton StateManager for embedding cache (shared across tabs)
- Per-tab session state classes (independent)
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from semantic_kinematics.mcp.state_manager import StateManager


# Singleton for embedding cache (shared across all tabs)
# Lazy initialization - model loads on first embedding request
state_manager = StateManager()


@dataclass
class DriftSession:
    """Session state for the Drift tab."""
    history: list[dict] = field(default_factory=list)


@dataclass
class TrajectorySession:
    """Session state for the Trajectory tab."""
    last_result: Optional[dict] = None
    last_comparison: Optional[dict] = None
    last_metrics: Optional[Any] = None  # TrajectoryMetrics (avoids heavy import)
    last_golden_metrics: Optional[Any] = None  # For compare tab reactive updates
    last_synthetic_metrics: Optional[Any] = None


# Per-tab session instances
drift_session = DriftSession()
trajectory_session = TrajectorySession()
