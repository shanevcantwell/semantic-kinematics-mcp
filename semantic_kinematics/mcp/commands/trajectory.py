"""
Trajectory analysis command module for text dynamics detection.

Tools:
- analyze_trajectory: Compute semantic trajectory metrics for a passage
- compare_trajectories: Compare two passages structurally
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from mcp.types import Tool

from semantic_kinematics.mcp.state_manager import StateManager


def get_tools() -> List[Tool]:
    """Return trajectory analysis tool definitions."""
    return [
        Tool(
            name="analyze_trajectory",
            description=(
                "Analyze semantic trajectory of a text passage. "
                "Treats text as a particle moving through embedding space, computing "
                "velocity, acceleration, and curvature. Returns deadpan_score (isolated "
                "acceleration spikes against calm background) and heller_score (circular "
                "repetitive structure with deceleration)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text passage to analyze (needs 2+ sentences)"
                    },
                    "acceleration_threshold": {
                        "type": "number",
                        "description": "Threshold for acceleration spikes (default: 0.3)",
                        "default": 0.3
                    },
                    "include_sentences": {
                        "type": "boolean",
                        "description": "Include sentence breakdown in output (default: false)",
                        "default": False
                    },
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="compare_trajectories",
            description=(
                "Compare two text passages structurally. "
                "Returns fitness_score (lower = better match). Evaluates both absolute "
                "quality (does the passage have strong trajectory structure?) and relative "
                "similarity (do the two passages share pacing rhythm?)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "golden_text": {
                        "type": "string",
                        "description": "Reference passage (the target structure)"
                    },
                    "synthetic_text": {
                        "type": "string",
                        "description": "Passage to compare against the reference"
                    },
                    "acceleration_threshold": {
                        "type": "number",
                        "description": "Threshold for acceleration spikes (default: 0.3)",
                        "default": 0.3
                    },
                },
                "required": ["golden_text", "synthetic_text"]
            }
        ),
    ]


# --- Internal dataclasses ---

@dataclass
class AccelerationSpike:
    """Details of a detected acceleration spike."""
    index: int
    magnitude: float
    isolation_score: float
    position_ratio: float

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "magnitude": round(self.magnitude, 4),
            "isolation_score": round(self.isolation_score, 4),
            "position_ratio": round(self.position_ratio, 4),
        }


@dataclass
class TrajectoryMetrics:
    """Complete trajectory analysis for a passage."""
    sentences: List[str]
    embeddings: np.ndarray

    velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    accelerations: np.ndarray = field(default_factory=lambda: np.array([]))
    curvatures: np.ndarray = field(default_factory=lambda: np.array([]))

    mean_velocity: float = 0.0
    velocity_variance: float = 0.0
    max_curvature: float = 0.0
    max_curvature_index: int = 0

    mean_acceleration: float = 0.0
    acceleration_variance: float = 0.0
    max_acceleration: float = 0.0
    max_acceleration_index: int = 0
    acceleration_spikes: List[AccelerationSpike] = field(default_factory=list)
    deadpan_score: float = 0.0

    circularity_score: float = 0.0
    tautology_density: float = 0.0
    deceleration_score: float = 0.0
    heller_score: float = 0.0

    def to_dict(self, include_sentences: bool = False) -> dict:
        """Convert to JSON-serializable dict."""
        result = {
            "n_sentences": len(self.sentences),
            "mean_velocity": round(self.mean_velocity, 4),
            "velocity_variance": round(self.velocity_variance, 6),
            "mean_acceleration": round(self.mean_acceleration, 4),
            "acceleration_variance": round(self.acceleration_variance, 6),
            "max_acceleration": round(self.max_acceleration, 4),
            "max_acceleration_index": self.max_acceleration_index,
            "max_curvature": round(self.max_curvature, 4),
            "max_curvature_index": self.max_curvature_index,
            "acceleration_spikes": [s.to_dict() for s in self.acceleration_spikes],
            "deadpan_score": round(self.deadpan_score, 4),
            "circularity_score": round(self.circularity_score, 4),
            "tautology_density": round(self.tautology_density, 4),
            "deceleration_score": round(self.deceleration_score, 4),
            "heller_score": round(self.heller_score, 4),
        }
        if include_sentences:
            result["sentences"] = self.sentences
            result["velocity_profile"] = [round(v, 4) for v in self.velocities.tolist()]
            result["acceleration_profile"] = [round(a, 4) for a in self.accelerations.tolist()]
            result["curvature_profile"] = [round(c, 4) for c in self.curvatures.tolist()]
        return result


class TrajectoryAnalyzer:
    """
    Computes semantic trajectory metrics for text passages.

    Key insight: In high-dimensional embedding space, curvature is uniformly high.
    The discriminating signal is in ACCELERATION — the rate of pacing change.
    Isolated acceleration spikes indicate sharp semantic shifts against steady background.
    """

    def __init__(
        self,
        manager: StateManager,
        acceleration_spike_threshold: float = 0.3,
    ):
        self._manager = manager
        self._nlp = None
        self.acceleration_spike_threshold = acceleration_spike_threshold

    def _get_nlp(self):
        """Lazy-load spaCy model."""
        if self._nlp is None:
            import spacy
            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                from spacy.cli import download
                download("en_core_web_sm")
                self._nlp = spacy.load("en_core_web_sm")
        return self._nlp

    def tokenize_sentences(self, text: str) -> List[str]:
        """Split text into sentences using spaCy."""
        nlp = self._get_nlp()
        doc = nlp(text)
        return [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    def embed_sentences(self, sentences: List[str]) -> np.ndarray:
        """Generate embeddings for each sentence."""
        adapter = self._manager.get_adapter()
        return adapter.embed_batch(sentences)

    def compute_velocities(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute semantic velocity between consecutive embeddings."""
        if len(embeddings) < 2:
            return np.array([])
        displacements = np.diff(embeddings, axis=0)
        return np.linalg.norm(displacements, axis=1)

    def compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute semantic acceleration (rate of velocity change)."""
        if len(velocities) < 2:
            return np.array([])
        return np.abs(np.diff(velocities))

    def compute_curvatures(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute angular curvature at each transition."""
        if len(embeddings) < 3:
            return np.array([])

        displacements = np.diff(embeddings, axis=0)
        curvatures = []

        for i in range(1, len(displacements)):
            d_prev = displacements[i - 1]
            d_curr = displacements[i]

            norm_prev = np.linalg.norm(d_prev)
            norm_curr = np.linalg.norm(d_curr)

            if norm_prev < 1e-10 or norm_curr < 1e-10:
                curvatures.append(0.0)
                continue

            cos_angle = np.dot(d_prev, d_curr) / (norm_prev * norm_curr)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            curvatures.append(float(np.arccos(cos_angle)))

        return np.array(curvatures)

    def detect_acceleration_spikes(self, accelerations: np.ndarray) -> List[AccelerationSpike]:
        """Detect acceleration spikes with isolation scores."""
        if len(accelerations) < 1:
            return []

        spikes = []
        threshold = self.acceleration_spike_threshold

        for i, a in enumerate(accelerations):
            if a >= threshold:
                neighbors = []
                if i > 0:
                    neighbors.append(accelerations[i - 1])
                if i < len(accelerations) - 1:
                    neighbors.append(accelerations[i + 1])

                if neighbors:
                    mean_neighbor = np.mean(neighbors)
                    isolation_ratio = (a - mean_neighbor) / (a + 0.01)
                    isolation_score = float(np.tanh(isolation_ratio * 2))
                else:
                    isolation_score = 0.5

                position_ratio = i / max(len(accelerations) - 1, 1)

                spikes.append(AccelerationSpike(
                    index=i,
                    magnitude=float(a),
                    isolation_score=max(0.0, isolation_score),
                    position_ratio=position_ratio,
                ))

        return spikes

    def compute_deadpan_score(
        self,
        accelerations: np.ndarray,
        spikes: List[AccelerationSpike],
    ) -> float:
        """Compute spike isolation score: isolated spikes against calm background."""
        if len(accelerations) < 2:
            return 0.0

        spike_ratio = len(spikes) / len(accelerations)
        spikiness_score = 1.0 - min(spike_ratio * 3, 1.0)

        mean_isolation = np.mean([s.isolation_score for s in spikes]) if spikes else 0.0

        non_spike_indices = [
            i for i in range(len(accelerations))
            if not any(s.index == i for s in spikes)
        ]
        if non_spike_indices:
            background_mean = np.mean([accelerations[i] for i in non_spike_indices])
            background_stability = 1.0 - min(background_mean / 0.3, 1.0)
        else:
            background_stability = 0.0

        if spikes and non_spike_indices:
            max_spike = max(s.magnitude for s in spikes)
            background_max = max([accelerations[i] for i in non_spike_indices], default=0.0)
            contrast = (max_spike - background_max) / (max_spike + 0.01)
            contrast_score = float(np.tanh(contrast * 2))
        else:
            contrast_score = 0.0

        deadpan = (
            0.25 * spikiness_score +
            0.35 * mean_isolation +
            0.20 * background_stability +
            0.20 * contrast_score
        )

        return float(np.clip(deadpan, 0.0, 1.0))

    def compute_circularity(self, embeddings: np.ndarray) -> float:
        """Detect circular/looping structure."""
        if len(embeddings) < 3:
            return 0.0

        loop_similarities = []
        for i in range(2, len(embeddings)):
            e_curr = embeddings[i]
            e_prev2 = embeddings[i - 2]

            norm_curr = np.linalg.norm(e_curr)
            norm_prev = np.linalg.norm(e_prev2)

            if norm_curr > 1e-10 and norm_prev > 1e-10:
                sim = np.dot(e_curr, e_prev2) / (norm_curr * norm_prev)
                loop_similarities.append(sim)

        if not loop_similarities:
            return 0.0

        mean_loop_sim = np.mean(loop_similarities)
        circularity = (mean_loop_sim - 0.3) / 0.5
        return float(np.clip(circularity, 0.0, 1.0))

    def compute_tautology_density(self, embeddings: np.ndarray, velocities: np.ndarray) -> float:
        """Detect tautological structure."""
        if len(embeddings) < 2:
            return 0.0

        pairwise_sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                e_i, e_j = embeddings[i], embeddings[j]
                norm_i, norm_j = np.linalg.norm(e_i), np.linalg.norm(e_j)
                if norm_i > 1e-10 and norm_j > 1e-10:
                    sim = np.dot(e_i, e_j) / (norm_i * norm_j)
                    pairwise_sims.append(sim)

        if not pairwise_sims:
            return 0.0

        mean_similarity = np.mean(pairwise_sims)
        similarity_score = (mean_similarity - 0.3) / 0.5

        if len(embeddings) >= 2:
            start, end = embeddings[0], embeddings[-1]
            norm_s, norm_e = np.linalg.norm(start), np.linalg.norm(end)
            if norm_s > 1e-10 and norm_e > 1e-10:
                net_displacement = np.linalg.norm(end - start)
                total_path = np.sum(velocities) if len(velocities) > 0 else 1.0
                displacement_ratio = net_displacement / (total_path + 0.01)
                displacement_score = 1.0 - min(displacement_ratio, 1.0)
            else:
                displacement_score = 0.0
        else:
            displacement_score = 0.0

        tautology = 0.6 * np.clip(similarity_score, 0.0, 1.0) + 0.4 * displacement_score
        return float(np.clip(tautology, 0.0, 1.0))

    def compute_deceleration(self, velocities: np.ndarray) -> float:
        """Detect deceleration pattern."""
        if len(velocities) < 2:
            return 0.0

        decelerations = sum(1 for i in range(1, len(velocities)) if velocities[i] < velocities[i - 1])
        decel_ratio = decelerations / (len(velocities) - 1)

        if len(velocities) >= 3:
            x = np.arange(len(velocities))
            slope = np.polyfit(x, velocities, 1)[0]
            trend_score = 1.0 if slope < -0.05 else (0.5 if slope < 0 else 0.0)
        else:
            trend_score = 0.5

        return float(np.clip(0.6 * decel_ratio + 0.4 * trend_score, 0.0, 1.0))

    def compute_heller_score(self, circularity: float, tautology: float, deceleration: float) -> float:
        """Combined circularity/repetition score."""
        heller = 0.35 * circularity + 0.40 * tautology + 0.25 * deceleration
        return float(np.clip(heller, 0.0, 1.0))

    def analyze(self, text: str) -> TrajectoryMetrics:
        """Perform full trajectory analysis on a text passage."""
        sentences = self.tokenize_sentences(text)

        if len(sentences) < 2:
            raise ValueError(f"Need at least 2 sentences, got {len(sentences)}")

        embeddings = self.embed_sentences(sentences)
        velocities = self.compute_velocities(embeddings)
        accelerations = self.compute_accelerations(velocities)
        curvatures = self.compute_curvatures(embeddings)

        accel_spikes = self.detect_acceleration_spikes(accelerations)
        deadpan = self.compute_deadpan_score(accelerations, accel_spikes)

        circularity = self.compute_circularity(embeddings)
        tautology = self.compute_tautology_density(embeddings, velocities)
        deceleration = self.compute_deceleration(velocities)
        heller = self.compute_heller_score(circularity, tautology, deceleration)

        return TrajectoryMetrics(
            sentences=sentences,
            embeddings=embeddings,
            velocities=velocities,
            accelerations=accelerations,
            curvatures=curvatures,
            mean_velocity=float(np.mean(velocities)) if len(velocities) > 0 else 0.0,
            velocity_variance=float(np.var(velocities)) if len(velocities) > 0 else 0.0,
            max_curvature=float(np.max(curvatures)) if len(curvatures) > 0 else 0.0,
            max_curvature_index=int(np.argmax(curvatures)) + 1 if len(curvatures) > 0 else 0,
            mean_acceleration=float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,
            acceleration_variance=float(np.var(accelerations)) if len(accelerations) > 0 else 0.0,
            max_acceleration=float(np.max(accelerations)) if len(accelerations) > 0 else 0.0,
            max_acceleration_index=int(np.argmax(accelerations)) if len(accelerations) > 0 else 0,
            acceleration_spikes=accel_spikes,
            deadpan_score=deadpan,
            circularity_score=circularity,
            tautology_density=tautology,
            deceleration_score=deceleration,
            heller_score=heller,
        )

    def compare(self, golden: TrajectoryMetrics, synthetic: TrajectoryMetrics) -> dict:
        """Compare two passages structurally."""
        result = {}

        # Absolute quality metrics
        result["synthetic_deadpan"] = round(synthetic.deadpan_score, 4)
        result["synthetic_heller"] = round(synthetic.heller_score, 4)

        if synthetic.acceleration_spikes:
            result["synthetic_mean_isolation"] = round(
                np.mean([s.isolation_score for s in synthetic.acceleration_spikes]), 4
            )
        else:
            result["synthetic_mean_isolation"] = 0.0

        # Relative similarity metrics
        if len(golden.accelerations) > 0 and len(synthetic.accelerations) > 0:
            result["acceleration_dtw"] = round(self._dtw(golden.accelerations, synthetic.accelerations), 4)
        else:
            result["acceleration_dtw"] = float("inf")

        if len(golden.accelerations) > 1 and len(synthetic.accelerations) > 1:
            len_a = len(golden.accelerations)
            len_b = len(synthetic.accelerations)
            target_len = max(len_a, len_b)

            a_interp = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len_a),
                golden.accelerations,
            )
            b_interp = np.interp(
                np.linspace(0, 1, target_len),
                np.linspace(0, 1, len_b),
                synthetic.accelerations,
            )

            if np.std(a_interp) > 1e-10 and np.std(b_interp) > 1e-10:
                result["acceleration_correlation"] = round(float(np.corrcoef(a_interp, b_interp)[0, 1]), 4)
            else:
                result["acceleration_correlation"] = 0.0
        else:
            result["acceleration_correlation"] = 0.0

        # Spike position matching
        if golden.acceleration_spikes and synthetic.acceleration_spikes:
            positions_a = [s.position_ratio for s in golden.acceleration_spikes]
            positions_b = [s.position_ratio for s in synthetic.acceleration_spikes]

            min_dist_sum = 0.0
            used_b = set()
            for pa in positions_a:
                best_dist = 1.0
                best_j = -1
                for j, pb in enumerate(positions_b):
                    if j not in used_b:
                        dist = abs(pa - pb)
                        if dist < best_dist:
                            best_dist = dist
                            best_j = j
                min_dist_sum += best_dist
                if best_j >= 0:
                    used_b.add(best_j)

            unmatched = abs(len(positions_a) - len(positions_b))
            avg_dist = min_dist_sum / max(len(positions_a), 1)
            result["spike_position_match"] = round(1.0 - min(avg_dist + unmatched * 0.2, 1.0), 4)
        elif not golden.acceleration_spikes and not synthetic.acceleration_spikes:
            result["spike_position_match"] = 1.0
        else:
            result["spike_position_match"] = 0.0

        # Spike count similarity
        n_golden = len(golden.acceleration_spikes)
        n_synthetic = len(synthetic.acceleration_spikes)
        if n_golden > 0:
            spike_count_ratio = min(n_synthetic, n_golden) / max(n_synthetic, n_golden, 1)
        else:
            spike_count_ratio = 1.0 if n_synthetic == 0 else 0.0
        result["spike_count_match"] = round(spike_count_ratio, 4)

        # Fitness score (lower = better)
        deadpan_penalty = 1.0 - result["synthetic_deadpan"]
        isolation_penalty = 1.0 - result["synthetic_mean_isolation"]
        rhythm_penalty = min(result["acceleration_dtw"], 1.0)
        correlation_penalty = 1.0 - max(result["acceleration_correlation"], 0)
        position_penalty = 1.0 - result["spike_position_match"]
        count_penalty = 1.0 - result["spike_count_match"]

        result["fitness_score"] = round(
            0.30 * deadpan_penalty +
            0.15 * isolation_penalty +
            0.20 * rhythm_penalty +
            0.15 * correlation_penalty +
            0.10 * position_penalty +
            0.10 * count_penalty,
            4
        )

        # Interpretation
        if result["fitness_score"] < 0.3:
            result["interpretation"] = "Excellent structural match"
        elif result["fitness_score"] < 0.5:
            result["interpretation"] = "Good match, some rhythm deviation"
        elif result["fitness_score"] < 0.7:
            result["interpretation"] = "Moderate — structure present but weak"
        else:
            result["interpretation"] = "Poor match"

        return result

    def _dtw(self, seq_a: np.ndarray, seq_b: np.ndarray) -> float:
        """Dynamic Time Warping distance (normalized)."""
        if len(seq_a) == 0 or len(seq_b) == 0:
            return float("inf")

        n, m = len(seq_a), len(seq_b)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq_a[i - 1] - seq_b[j - 1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )

        path_len = n + m - 1
        return dtw_matrix[n, m] / path_len if path_len > 0 else 0.0


# --- Tool handlers ---

async def analyze_trajectory(
    manager: StateManager,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Analyze semantic trajectory of a text passage.

    Returns spike isolation and circularity scores, plus detailed metrics.
    """
    text = args.get("text", "")
    threshold = args.get("acceleration_threshold", 0.3)
    include_sentences = args.get("include_sentences", False)

    if not text:
        return {"error": "No text provided"}

    try:
        analyzer = TrajectoryAnalyzer(manager, acceleration_spike_threshold=threshold)
        metrics = analyzer.analyze(text)

        result = metrics.to_dict(include_sentences=include_sentences)

        # Add interpretation
        if metrics.deadpan_score > 0.6:
            result["spike_interpretation"] = "Strong isolated spikes against steady background"
        elif metrics.deadpan_score > 0.4:
            result["spike_interpretation"] = "Moderate isolated spike pattern"
        else:
            result["spike_interpretation"] = "Weak or no isolated spike pattern"

        if metrics.heller_score > 0.5:
            result["circularity_interpretation"] = "Strong circular/repetitive structure with deceleration"
        elif metrics.heller_score > 0.3:
            result["circularity_interpretation"] = "Moderate circular structure"
        else:
            result["circularity_interpretation"] = "Weak or no circular structure"

        return result

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Analysis failed: {str(e)}"}


async def compare_trajectories_handler(
    manager: StateManager,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare two passages structurally.

    Returns fitness_score (lower = better match) and component metrics.
    """
    golden_text = args.get("golden_text", "")
    synthetic_text = args.get("synthetic_text", "")
    threshold = args.get("acceleration_threshold", 0.3)

    if not golden_text:
        return {"error": "No golden_text provided"}
    if not synthetic_text:
        return {"error": "No synthetic_text provided"}

    try:
        analyzer = TrajectoryAnalyzer(manager, acceleration_spike_threshold=threshold)

        golden_metrics = analyzer.analyze(golden_text)
        synthetic_metrics = analyzer.analyze(synthetic_text)

        comparison = analyzer.compare(golden_metrics, synthetic_metrics)

        comparison["reference_summary"] = {
            "n_sentences": len(golden_metrics.sentences),
            "deadpan_score": round(golden_metrics.deadpan_score, 4),
            "heller_score": round(golden_metrics.heller_score, 4),
            "n_spikes": len(golden_metrics.acceleration_spikes),
        }
        comparison["comparison_summary"] = {
            "n_sentences": len(synthetic_metrics.sentences),
            "deadpan_score": round(synthetic_metrics.deadpan_score, 4),
            "heller_score": round(synthetic_metrics.heller_score, 4),
            "n_spikes": len(synthetic_metrics.acceleration_spikes),
        }

        return comparison

    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Comparison failed: {str(e)}"}
