"""
Tests for the precomputed-matrix and pre-segmented seams on TrajectoryAnalyzer.

These exercise ``analyze_embeddings`` (no embedder needed at all) and
``analyze_segments`` (embed path monkeypatched to canned vectors), confirming:

- a synthetic flat-then-jump matrix puts ``max_acceleration_index`` at the
  engineered jump and the spike detector flags it;
- ``analyze_segments`` does NOT sentence-split -- it embeds exactly the list it
  is given, so the step count tracks ``len(segments)``, not sentence count.

No spaCy, no live embedding backend -> CI-safe.
"""

import numpy as np
import pytest

from semantic_kinematics.mcp.commands.trajectory import TrajectoryAnalyzer
from semantic_kinematics.mcp.state_manager import StateManager


def _make_analyzer(threshold=0.3):
    # StateManager is constructed but never touched: analyze_embeddings needs no
    # adapter, and analyze_segments has its embed path monkeypatched per-test.
    return TrajectoryAnalyzer(StateManager(), acceleration_spike_threshold=threshold)


def test_analyze_embeddings_flags_engineered_jump():
    """A flat run of steps followed by one sharp jump must surface as the peak.

    Build a 1-D-walk-in-N-d trajectory: the point creeps along the first axis in
    small equal steps (flat velocity -> ~zero acceleration), then takes one large
    step. The single velocity change at the jump is the only acceleration event,
    so ``max_acceleration_index`` must land on it and the spike detector (whose
    threshold we sit comfortably below) must flag exactly that index.
    """
    dim = 8
    # 6 calm steps of size 0.05, then one jump of size 1.0, then 5 more calm.
    step_sizes = [0.05] * 6 + [1.0] + [0.05] * 5
    positions = [np.zeros(dim)]
    for s in step_sizes:
        nxt = positions[-1].copy()
        nxt[0] += s
        positions.append(nxt)
    embeddings = np.array(positions)  # [n_steps, dim]

    analyzer = _make_analyzer(threshold=0.3)
    metrics = analyzer.analyze_embeddings(embeddings)

    # velocities length = n_steps - 1; accelerations length = n_steps - 2.
    # The jump is step index 6 (0-based in step_sizes). velocity[6] is the big
    # one; |v[6]-v[5]| is the rising edge -> acceleration index 5.
    accels = metrics.accelerations
    expected_peak = 5
    assert metrics.max_acceleration_index == expected_peak, (
        f"peak at {metrics.max_acceleration_index}, profile={accels.tolist()}"
    )
    # The jump dwarfs the flat background.
    assert metrics.max_acceleration > 0.5
    # The spike detector flagged the engineered jump (rising and falling edges
    # both clear the threshold; the rising edge at index 5 must be among them).
    spike_indices = {s.index for s in metrics.acceleration_spikes}
    assert expected_peak in spike_indices, f"spikes at {sorted(spike_indices)}"


def test_analyze_embeddings_labels_ride_and_validate():
    """Labels ride alongside as ``sentences``; length mismatch is rejected."""
    embeddings = np.eye(4)  # 4 steps, 4 dims
    labels = ["a", "b", "c", "d"]
    metrics = _make_analyzer().analyze_embeddings(embeddings, labels=labels)
    assert metrics.sentences == labels
    # The label at the peak index is addressable.
    _ = metrics.sentences[metrics.max_acceleration_index]

    with pytest.raises(ValueError):
        _make_analyzer().analyze_embeddings(embeddings, labels=["too", "few"])


def test_analyze_embeddings_rejects_non_2d():
    with pytest.raises(ValueError):
        _make_analyzer().analyze_embeddings(np.zeros(5))  # 1-D


def test_analyze_segments_does_not_sentence_split(monkeypatch):
    """Three MULTI-sentence strings must produce exactly three trajectory steps.

    If ``analyze_segments`` were (wrongly) sentence-splitting, the canned
    embedder below would be asked to embed far more than 3 items and the step
    count would balloon. We monkeypatch ``embed_sentences`` to (a) record what
    it was handed and (b) return one vector per input element. The assertion is
    that it received exactly the 3 segments verbatim -> no splitting occurred.
    """
    segments = [
        "First sentence here. Second sentence in the same segment.",
        "Another segment. With two sentences. Actually three.",
        "Final one. Also multi. Sentence.",
    ]

    captured = {}

    def fake_embed(self, texts):
        captured["texts"] = list(texts)
        # One distinct unit vector per input element.
        n = len(texts)
        mat = np.zeros((n, 6))
        for i in range(n):
            mat[i, i % 6] = 1.0
        return mat

    monkeypatch.setattr(TrajectoryAnalyzer, "embed_sentences", fake_embed)

    metrics = _make_analyzer().analyze_segments(segments)

    # The embed path saw exactly the segments, unsplit.
    assert captured["texts"] == segments
    # n_sentences (the step labels) equals len(segments), NOT total sentence
    # count (which would be 3+3+3 = 9 if it had split).
    assert len(metrics.sentences) == len(segments)
    assert metrics.sentences == segments
    # Derivative lengths follow from 3 steps: 2 velocities, 1 acceleration.
    assert len(metrics.velocities) == len(segments) - 1


def test_analyze_segments_requires_two():
    with pytest.raises(ValueError):
        _make_analyzer().analyze_segments(["only one"])
