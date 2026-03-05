"""
Event handlers for the Trajectory tab.

Analyzes text structure via semantic trajectory metrics.
Supports single passage analysis and comparison mode.
"""

import json
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from semantic_kinematics.mcp.commands.trajectory import (
    TrajectoryAnalyzer,
    TrajectoryMetrics,
    analyze_trajectory,
    compare_trajectories_handler,
)
from semantic_kinematics.mcp.commands.model import (
    model_status as _model_status,
    model_load as _model_load,
    model_unload as _model_unload,
)
from semantic_kinematics.ui.state import state_manager, trajectory_session


async def analyze_single(
    text: str,
    threshold: float,
    window_size: float = 1,
) -> tuple[float | None, float | None, str, str, dict | None, Any, Any, Any, str]:
    """
    Analyze a single passage for structural patterns.

    Uses TrajectoryAnalyzer directly (bypasses MCP serialization) to retain
    full embeddings for PCA and heatmap visualizations.

    Embeddings are computed once and cached. The window_size and threshold
    parameters are applied on the cached embeddings via _recompute_and_plot().

    Returns:
        (deadpan_score, heller_score, spike_interp, circularity_interp,
         spikes_json, trajectory_plot, pca_plot, heatmap_plot, status)
    """
    empty = (None, None, "", "", None, None, None, None)

    if not text or not text.strip():
        return *empty, "No text provided"

    try:
        analyzer = TrajectoryAnalyzer(state_manager, acceleration_spike_threshold=threshold)
        metrics = analyzer.analyze(text)
    except ValueError as e:
        return *empty, f"Error: {e}"
    except Exception as e:
        return *empty, f"Error: {e}"

    # Cache raw metrics (full embeddings) for reactive slider updates
    trajectory_session.last_result = metrics.to_dict(include_sentences=True)
    trajectory_session.last_metrics = metrics

    # Apply smoothing and threshold via the shared recompute path
    return _recompute_and_plot(threshold, window_size)


def _build_trajectory_plot(result: dict, threshold: float) -> go.Figure:
    """Build velocity/acceleration/curvature profile plot from analysis result."""
    velocities = result.get("velocity_profile", [])
    accelerations = result.get("acceleration_profile", [])
    curvatures = result.get("curvature_profile", [])
    spikes = result.get("acceleration_spikes", [])
    sentences = result.get("sentences", [])

    n_rows = 3 if curvatures else 2
    titles = ["Velocity (semantic pacing)", "Acceleration (pacing change)"]
    heights = [0.5, 0.5]
    if curvatures:
        titles.append("Curvature (direction change)")
        heights = [0.33, 0.33, 0.33]

    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=titles,
        row_heights=heights,
    )

    # Velocity profile
    vel_x = list(range(len(velocities)))
    fig.add_trace(
        go.Scatter(
            x=vel_x, y=velocities,
            mode="lines+markers",
            name="velocity",
            line=dict(color="#4A90D9", width=2),
            marker=dict(size=5),
            hovertext=[
                f"v={v:.3f}<br>{sentences[i+1][:60]}..." if i + 1 < len(sentences) else f"v={v:.3f}"
                for i, v in enumerate(velocities)
            ],
            hoverinfo="text",
        ),
        row=1, col=1,
    )

    # Acceleration profile
    acc_x = list(range(len(accelerations)))
    fig.add_trace(
        go.Scatter(
            x=acc_x, y=accelerations,
            mode="lines+markers",
            name="acceleration",
            line=dict(color="#7B8794", width=2),
            marker=dict(size=5),
        ),
        row=2, col=1,
    )

    # Spike threshold line
    if accelerations:
        fig.add_hline(
            y=threshold, row=2, col=1,
            line_dash="dash", line_color="#E74C3C", line_width=1,
            annotation_text=f"threshold={threshold}",
            annotation_position="top right",
            annotation_font_color="#E74C3C",
        )

    # Mark spikes
    for spike in spikes:
        idx = spike["index"]
        mag = spike["magnitude"]
        iso = spike["isolation_score"]
        fig.add_trace(
            go.Scatter(
                x=[idx], y=[mag],
                mode="markers",
                name=f"spike (iso={iso:.2f})",
                marker=dict(size=12, color="#E74C3C", symbol="diamond"),
                hovertext=f"SPIKE idx={idx}<br>a={mag:.3f}<br>isolation={iso:.3f}",
                hoverinfo="text",
                showlegend=False,
            ),
            row=2, col=1,
        )

    # Curvature profile
    if curvatures:
        curv_x = list(range(len(curvatures)))
        fig.add_trace(
            go.Scatter(
                x=curv_x, y=curvatures,
                mode="lines+markers",
                name="curvature",
                line=dict(color="#8E44AD", width=2),
                marker=dict(size=5),
                hovertext=[
                    f"\u03ba={c:.3f}<br>{sentences[i+1][:60]}..."
                    if i + 1 < len(sentences) else f"\u03ba={c:.3f}"
                    for i, c in enumerate(curvatures)
                ],
                hoverinfo="text",
            ),
            row=3, col=1,
        )

    fig.update_layout(
        height=620 if curvatures else 450,
        margin=dict(l=50, r=20, t=40, b=30),
        showlegend=False,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="sentence transition", row=n_rows, col=1)
    fig.update_yaxes(title_text="velocity", row=1, col=1)
    fig.update_yaxes(title_text="acceleration", row=2, col=1)
    if curvatures:
        fig.update_yaxes(title_text="curvature (rad)", row=3, col=1)

    return fig


def _compute_pca_2d(
    embeddings: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Project embeddings from high-D to 2D via PCA (numpy SVD).

    Returns:
        (coords_2d, explained_var_ratio_pc1, explained_var_ratio_pc2)
    """
    X = embeddings.astype(np.float64)
    X_centered = X - X.mean(axis=0)

    U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)

    coords_2d = U[:, :2] * S[:2]

    total_var = np.sum(S ** 2)
    if total_var > 0:
        ev1 = float(S[0] ** 2 / total_var)
        ev2 = float(S[1] ** 2 / total_var)
    else:
        ev1, ev2 = 0.0, 0.0

    return coords_2d, ev1, ev2


def _build_pca_plot(metrics: TrajectoryMetrics) -> go.Figure:
    """Build PCA 2D trajectory plot from full TrajectoryMetrics."""
    embeddings = metrics.embeddings
    sentences = metrics.sentences
    spikes = metrics.acceleration_spikes

    if len(embeddings) < 3:
        fig = go.Figure()
        fig.add_annotation(text="Need 3+ sentences for PCA plot", showarrow=False)
        return fig

    coords, ev1, ev2 = _compute_pca_2d(embeddings)
    n = len(coords)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords[:, 0].tolist(),
        y=coords[:, 1].tolist(),
        mode="lines+markers",
        marker=dict(
            size=7,
            color=list(range(n)),
            colorscale="Viridis",
            colorbar=dict(title="Sentence #", thickness=15),
            showscale=True,
        ),
        line=dict(color="rgba(100, 100, 100, 0.3)", width=1),
        hovertext=[
            f"#{i}: {s[:80]}..." if len(s) > 80 else f"#{i}: {s}"
            for i, s in enumerate(sentences)
        ],
        hoverinfo="text",
        name="trajectory",
    ))

    # Start / end markers
    fig.add_trace(go.Scatter(
        x=[coords[0, 0]], y=[coords[0, 1]],
        mode="markers+text",
        marker=dict(size=14, color="#27AE60", symbol="star"),
        text=["START"], textposition="top center",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=[coords[-1, 0]], y=[coords[-1, 1]],
        mode="markers+text",
        marker=dict(size=14, color="#E74C3C", symbol="square"),
        text=["END"], textposition="top center",
        showlegend=False,
    ))

    # Mark spike locations
    spike_indices = [s.index + 1 for s in spikes if s.index + 1 < n]
    if spike_indices:
        fig.add_trace(go.Scatter(
            x=coords[spike_indices, 0].tolist(),
            y=coords[spike_indices, 1].tolist(),
            mode="markers",
            marker=dict(
                size=10, color="#E74C3C", symbol="diamond",
                line=dict(width=1, color="white"),
            ),
            name="accel. spikes",
            hovertext=[f"SPIKE at sentence {i}" for i in spike_indices],
            hoverinfo="text",
        ))

    fig.update_layout(
        title=f"PCA 2D Trajectory (PC1: {ev1:.1%}, PC2: {ev2:.1%})",
        xaxis_title=f"PC1 ({ev1:.1%})",
        yaxis_title=f"PC2 ({ev2:.1%})",
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def _build_heatmap_plot(metrics: TrajectoryMetrics) -> go.Figure:
    """Build NxN cosine similarity heatmap from sentence embeddings."""
    embeddings = metrics.embeddings
    n = len(embeddings)

    if n < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need 2+ sentences for heatmap", showarrow=False)
        return fig

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    normed = embeddings / norms
    sim_matrix = (normed @ normed.T).astype(np.float64)

    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix.tolist(),
        x=list(range(n)),
        y=list(range(n)),
        colorscale="RdBu",
        zmid=0.5,
        zmin=0.0,
        zmax=1.0,
        colorbar=dict(title="Cosine Sim"),
        hovertemplate="Sentence %{x} vs %{y}<br>Similarity: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=f"Pairwise Cosine Similarity ({n} sentences)",
        xaxis_title="Sentence index",
        yaxis_title="Sentence index",
        height=500,
        template="plotly_white",
        margin=dict(l=50, r=20, t=50, b=40),
        yaxis=dict(autorange="reversed"),
    )

    return fig


def _smooth_embeddings(
    embeddings: np.ndarray,
    sentences: list[str],
    window_size: int,
) -> tuple[np.ndarray, list[str]]:
    """
    Average consecutive sentence embeddings with a sliding window.

    For window_size w, position i = mean(embeddings[i:i+w]).
    Produces (N - w + 1) smoothed positions.
    """
    if window_size <= 1 or len(embeddings) <= window_size:
        return embeddings, sentences

    n = len(embeddings)
    n_smooth = n - window_size + 1
    smoothed = np.zeros((n_smooth, embeddings.shape[1]), dtype=embeddings.dtype)

    cumsum = np.cumsum(embeddings, axis=0)
    smoothed[0] = cumsum[window_size - 1] / window_size
    smoothed[1:] = (cumsum[window_size:] - cumsum[:n_smooth - 1]) / window_size

    labels = [
        f"{sentences[i][:50]}... (+{window_size - 1})" if window_size > 1 else sentences[i]
        for i in range(n_smooth)
    ]

    return smoothed, labels


def _recompute_and_plot(
    threshold: float,
    window_size: int,
) -> tuple[float | None, float | None, str, str, dict | None, object, object, object, str]:
    """
    Recompute trajectory metrics and plots from cached embeddings.

    Uses trajectory_session.last_metrics (raw embeddings) to apply smoothing
    and threshold without re-running the embedding model.
    """
    empty = (None, None, "", "", None, None, None, None)
    raw_metrics = trajectory_session.last_metrics
    if raw_metrics is None:
        return *empty, "No analysis to update -- run Analyze first"

    try:
        embeddings = raw_metrics.embeddings
        sentences = raw_metrics.sentences

        smooth_emb, smooth_sent = _smooth_embeddings(
            embeddings, sentences, int(window_size)
        )

        if len(smooth_emb) < 3:
            return *empty, f"Window too large for {len(sentences)} sentences"

        analyzer = TrajectoryAnalyzer(
            state_manager, acceleration_spike_threshold=threshold
        )
        velocities = analyzer.compute_velocities(smooth_emb)
        accelerations = analyzer.compute_accelerations(velocities)
        curvatures = analyzer.compute_curvatures(smooth_emb)
        spikes = analyzer.detect_acceleration_spikes(accelerations)

        deadpan = analyzer.compute_deadpan_score(accelerations, spikes)
        circularity = analyzer.compute_circularity(smooth_emb)
        tautology = analyzer.compute_tautology_density(smooth_emb, velocities)
        deceleration = analyzer.compute_deceleration(velocities)
        heller = analyzer.compute_heller_score(circularity, tautology, deceleration)

        result = {
            "velocity_profile": velocities.tolist(),
            "acceleration_profile": accelerations.tolist(),
            "curvature_profile": curvatures.tolist(),
            "acceleration_spikes": [s.to_dict() for s in spikes],
            "sentences": smooth_sent,
        }

        class _SmoothMetrics:
            pass
        sm = _SmoothMetrics()
        sm.embeddings = smooth_emb
        sm.sentences = smooth_sent
        sm.acceleration_spikes = spikes

        trajectory_fig = _build_trajectory_plot(result, threshold)
        pca_fig = _build_pca_plot(sm)
        heatmap_fig = _build_heatmap_plot(sm)

        # Interpretations
        if deadpan > 0.6:
            spike_interp = "Strong isolated spikes against steady background"
        elif deadpan > 0.4:
            spike_interp = "Moderate isolated spike pattern"
        else:
            spike_interp = "Weak or no isolated spike pattern"

        if heller > 0.5:
            circ_interp = "Strong circular/repetitive structure with deceleration"
        elif heller > 0.3:
            circ_interp = "Moderate circular structure"
        else:
            circ_interp = "Weak or no circular structure"

        window_note = f" (window={int(window_size)})" if int(window_size) > 1 else ""
        return (
            deadpan,
            heller,
            spike_interp,
            circ_interp,
            {"spikes": result["acceleration_spikes"]},
            trajectory_fig,
            pca_fig,
            heatmap_fig,
            f"Recomputed {len(smooth_sent)} positions{window_note}"
        )

    except Exception as e:
        return *empty, f"Recompute error: {e}"


async def compare_passages(
    golden_text: str,
    synthetic_text: str,
    threshold: float,
    window_size: float = 1,
) -> tuple[float | None, str, dict | None, Any, str]:
    """
    Compare two passages structurally.

    Uses TrajectoryAnalyzer directly to retain full metrics for visualization.
    Caches raw metrics for reactive slider updates.
    """
    empty = (None, "", None, None)

    if not golden_text or not golden_text.strip():
        return *empty, "Passage A is required"

    if not synthetic_text or not synthetic_text.strip():
        return *empty, "Passage B is required"

    try:
        analyzer = TrajectoryAnalyzer(state_manager, acceleration_spike_threshold=threshold)
        golden_metrics = analyzer.analyze(golden_text)
        synthetic_metrics = analyzer.analyze(synthetic_text)
    except ValueError as e:
        return *empty, f"Error: {e}"
    except Exception as e:
        return *empty, f"Error: {e}"

    trajectory_session.last_golden_metrics = golden_metrics
    trajectory_session.last_synthetic_metrics = synthetic_metrics

    return _recompute_comparison(threshold, window_size)


def _recompute_comparison(
    threshold: float,
    window_size: float = 1,
) -> tuple[float | None, str, dict | None, Any, str]:
    """
    Recompute comparison from cached embeddings.

    Applies smoothing and threshold without re-running the embedding model.
    """
    empty = (None, "", None, None)
    golden_raw = trajectory_session.last_golden_metrics
    synthetic_raw = trajectory_session.last_synthetic_metrics

    if golden_raw is None or synthetic_raw is None:
        return *empty, "No comparison to update -- run Compare first"

    try:
        w = int(window_size)

        g_emb, g_sent = _smooth_embeddings(
            golden_raw.embeddings, golden_raw.sentences, w
        )
        s_emb, s_sent = _smooth_embeddings(
            synthetic_raw.embeddings, synthetic_raw.sentences, w
        )

        if len(g_emb) < 3 or len(s_emb) < 3:
            return *empty, f"Window too large for passage lengths"

        analyzer = TrajectoryAnalyzer(
            state_manager, acceleration_spike_threshold=threshold
        )

        # Passage A
        g_vel = analyzer.compute_velocities(g_emb)
        g_acc = analyzer.compute_accelerations(g_vel)
        g_spikes = analyzer.detect_acceleration_spikes(g_acc)
        g_deadpan = analyzer.compute_deadpan_score(g_acc, g_spikes)
        g_circ = analyzer.compute_circularity(g_emb)
        g_taut = analyzer.compute_tautology_density(g_emb, g_vel)
        g_decel = analyzer.compute_deceleration(g_vel)
        g_heller = analyzer.compute_heller_score(g_circ, g_taut, g_decel)

        # Passage B
        s_vel = analyzer.compute_velocities(s_emb)
        s_acc = analyzer.compute_accelerations(s_vel)
        s_spikes = analyzer.detect_acceleration_spikes(s_acc)
        s_deadpan = analyzer.compute_deadpan_score(s_acc, s_spikes)
        s_circ = analyzer.compute_circularity(s_emb)
        s_taut = analyzer.compute_tautology_density(s_emb, s_vel)
        s_decel = analyzer.compute_deceleration(s_vel)
        s_heller = analyzer.compute_heller_score(s_circ, s_taut, s_decel)

        class _M:
            pass
        gm, sm_ = _M(), _M()
        gm.velocities, gm.accelerations, gm.acceleration_spikes = g_vel, g_acc, g_spikes
        sm_.velocities, sm_.accelerations, sm_.acceleration_spikes = s_vel, s_acc, s_spikes

        details = {
            "passage_a": {
                "n_sentences": len(g_sent),
                "spike_isolation": round(g_deadpan, 4),
                "circularity": round(g_heller, 4),
                "n_spikes": len(g_spikes),
            },
            "passage_b": {
                "n_sentences": len(s_sent),
                "spike_isolation": round(s_deadpan, 4),
                "circularity": round(s_heller, 4),
                "n_spikes": len(s_spikes),
            },
        }

        comparison_fig = _build_comparison_plot(gm, sm_, threshold)

        window_note = f" (window={w})" if w > 1 else ""
        return (
            None,  # fitness_score not computed in recompute path
            "",
            details,
            comparison_fig,
            f"Comparison updated{window_note}"
        )

    except Exception as e:
        return *empty, f"Recompute error: {e}"


def _build_comparison_plot(
    golden: TrajectoryMetrics,
    synthetic: TrajectoryMetrics,
    threshold: float,
) -> go.Figure:
    """Build overlaid velocity/acceleration profiles for two passages."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        subplot_titles=("Velocity (overlaid)", "Acceleration (overlaid)"),
    )

    def norm_x(n: int) -> list[float]:
        """Normalize x-axis to [0, 1] so different-length texts align."""
        return np.linspace(0, 1, n).tolist() if n > 0 else []

    # Passage A velocity
    gv = golden.velocities
    fig.add_trace(go.Scatter(
        x=norm_x(len(gv)), y=gv.tolist(),
        mode="lines", name="A velocity",
        line=dict(color="#F39C12", width=2),
    ), row=1, col=1)

    # Passage B velocity
    sv = synthetic.velocities
    fig.add_trace(go.Scatter(
        x=norm_x(len(sv)), y=sv.tolist(),
        mode="lines", name="B velocity",
        line=dict(color="#3498DB", width=2),
    ), row=1, col=1)

    # Passage A acceleration
    ga = golden.accelerations
    fig.add_trace(go.Scatter(
        x=norm_x(len(ga)), y=ga.tolist(),
        mode="lines", name="A acceleration",
        line=dict(color="#F39C12", width=2),
    ), row=2, col=1)

    # Passage B acceleration
    sa = synthetic.accelerations
    fig.add_trace(go.Scatter(
        x=norm_x(len(sa)), y=sa.tolist(),
        mode="lines", name="B acceleration",
        line=dict(color="#3498DB", width=2),
    ), row=2, col=1)

    # Threshold line
    fig.add_hline(
        y=threshold, row=2, col=1,
        line_dash="dash", line_color="#E74C3C", line_width=1,
    )

    # A spike markers
    for spike in golden.acceleration_spikes:
        fig.add_trace(go.Scatter(
            x=[spike.position_ratio], y=[spike.magnitude],
            mode="markers",
            marker=dict(size=8, color="#F39C12", symbol="diamond"),
            showlegend=False,
            hovertext=f"A spike pos={spike.position_ratio:.2f} mag={spike.magnitude:.3f}",
            hoverinfo="text",
        ), row=2, col=1)

    # B spike markers
    for spike in synthetic.acceleration_spikes:
        fig.add_trace(go.Scatter(
            x=[spike.position_ratio], y=[spike.magnitude],
            mode="markers",
            marker=dict(size=8, color="#3498DB", symbol="diamond"),
            showlegend=False,
            hovertext=f"B spike pos={spike.position_ratio:.2f} mag={spike.magnitude:.3f}",
            hoverinfo="text",
        ), row=2, col=1)

    fig.update_layout(
        height=450,
        margin=dict(l=50, r=20, t=40, b=30),
        template="plotly_white",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )
    fig.update_xaxes(title_text="position (normalized)", row=2, col=1)
    fig.update_yaxes(title_text="velocity", row=1, col=1)
    fig.update_yaxes(title_text="acceleration", row=2, col=1)

    return fig


# --- Model management handlers ---

async def get_model_status() -> str:
    """Get current model status as formatted string."""
    result = await _model_status(state_manager, {})
    backend = result.get("backend", "unknown")
    model_name = result.get("model_name", "not initialized")
    loaded = result.get("is_loaded", False)
    cache = result.get("cache_size", 0)
    dims = result.get("dimensions", "?")
    status = "loaded" if loaded else "not loaded"
    return f"{backend} | {model_name} | {status} | {dims}d | cache: {cache}"


async def do_model_load(backend: str, server_url: str) -> str:
    """Load model with given backend config."""
    kwargs = {"backend": backend}
    if server_url and server_url.strip():
        kwargs["base_url"] = server_url.strip()
    result = await _model_load(state_manager, kwargs)
    if "error" in result:
        return f"Error: {result['error']}"
    return f"Loaded: {result.get('model_name', '?')} ({result.get('backend', '?')})"


async def do_model_unload() -> str:
    """Unload model and clear cache."""
    result = await _model_unload(state_manager, {"clear_cache": True})
    cleared = result.get("cache_entries_cleared", 0)
    return f"{result.get('status', 'done')} | {cleared} cache entries cleared"
