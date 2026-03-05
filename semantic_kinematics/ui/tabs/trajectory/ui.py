"""
UI layout for the Trajectory tab.

Analyzes text structure via semantic trajectory metrics.
Supports single passage analysis and two-passage comparison.
Includes model load/unload controls and trajectory profile plots.
"""

import gradio as gr

from semantic_kinematics.ui.tabs.trajectory.handlers import (
    analyze_single,
    compare_passages,
    get_model_status,
    do_model_load,
    do_model_unload,
    _recompute_and_plot,
    _recompute_comparison,
)


def build_tab():
    """Build the Trajectory tab UI components and wire up events."""

    gr.Markdown("""
    ## Trajectory: Text Dynamics Analysis

    Analyzes text as a particle moving through embedding space.
    Each sentence is a point; metrics describe the path between them.

    | Metric | What it measures |
    |--------|-----------------|
    | **Velocity** | Semantic distance between consecutive sentences |
    | **Acceleration** | Rate of change in pacing |
    | **Curvature** | Directional change in embedding space |
    | **Circularity** | Semantic looping (sentence i resembles sentence i-2) |
    | **Tautology density** | High pairwise similarity with low net displacement |
    """)

    # Model controls
    with gr.Accordion("Embedding Model", open=False):
        model_status_box = gr.Textbox(
            label="Status",
            interactive=False,
            value="(click Refresh)"
        )

        with gr.Row():
            backend_dropdown = gr.Dropdown(
                choices=["nv_embed", "lmstudio", "sentence_transformers"],
                value="nv_embed",
                label="Backend",
                scale=1,
            )
            server_url = gr.Textbox(
                label="Server URL (lmstudio only)",
                value="http://localhost:1234/v1",
                scale=2,
            )

        with gr.Row():
            refresh_btn = gr.Button("Refresh", size="sm")
            load_btn = gr.Button("Load", variant="primary", size="sm")
            unload_btn = gr.Button("Unload", variant="stop", size="sm")

        refresh_btn.click(fn=get_model_status, outputs=[model_status_box])
        load_btn.click(fn=do_model_load, inputs=[backend_dropdown, server_url], outputs=[model_status_box])
        unload_btn.click(fn=do_model_unload, outputs=[model_status_box])

    with gr.Tabs():
        # Single passage analysis
        with gr.Tab("Analyze Single"):
            text_input = gr.Textbox(
                label="Passage",
                placeholder="Enter text passage (needs 2+ sentences)...",
                lines=8
            )

            with gr.Accordion("Analysis Controls", open=True):
                threshold_single = gr.Slider(
                    minimum=0.1,
                    maximum=0.6,
                    value=0.3,
                    step=0.05,
                    label="Acceleration threshold",
                )
                window_single = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="Context window",
                    info="Sentences to average (1 = no smoothing)",
                )

            analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Row():
                deadpan_score = gr.Number(
                    label="Spike Isolation Score",
                    precision=4,
                    interactive=False
                )
                heller_score = gr.Number(
                    label="Circularity Score",
                    precision=4,
                    interactive=False
                )

            with gr.Row():
                adams_interp = gr.Textbox(
                    label="Spike Pattern",
                    interactive=False
                )
                heller_interp = gr.Textbox(
                    label="Circularity Pattern",
                    interactive=False
                )

            trajectory_plot = gr.Plot(
                label="Trajectory Profile",
            )

            with gr.Row():
                pca_plot = gr.Plot(
                    label="PCA 2D Trajectory",
                )
                heatmap_plot = gr.Plot(
                    label="Cosine Similarity Heatmap",
                )

            spikes_json = gr.JSON(
                label="Acceleration Spikes",
                visible=True
            )

            single_status = gr.Textbox(
                label="Status",
                interactive=False
            )

            # Wire up analyze button
            single_outputs = [
                deadpan_score, heller_score, adams_interp, heller_interp,
                spikes_json, trajectory_plot, pca_plot, heatmap_plot,
                single_status,
            ]
            analyze_btn.click(
                fn=analyze_single,
                inputs=[text_input, threshold_single, window_single],
                outputs=single_outputs,
            )

            # Reactive slider updates (recompute from cached embeddings)
            for slider in [threshold_single, window_single]:
                slider.change(
                    fn=_recompute_and_plot,
                    inputs=[threshold_single, window_single],
                    outputs=single_outputs,
                )

        # Comparison mode
        with gr.Tab("Compare Two Passages"):
            gr.Markdown("""
            Compare two passages structurally.
            **Fitness score:** Lower = closer structural match.
            """)

            with gr.Row():
                golden_input = gr.Textbox(
                    label="Passage A (reference)",
                    placeholder="Paste reference passage...",
                    lines=6
                )
                synthetic_input = gr.Textbox(
                    label="Passage B (to compare)",
                    placeholder="Paste passage to compare...",
                    lines=6
                )

            with gr.Accordion("Analysis Controls", open=True):
                threshold_compare = gr.Slider(
                    minimum=0.1,
                    maximum=0.6,
                    value=0.3,
                    step=0.05,
                    label="Acceleration threshold",
                )
                window_compare = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=1,
                    step=1,
                    label="Context window",
                    info="Sentences to average (1 = no smoothing)",
                )

            compare_btn = gr.Button("Compare", variant="primary")

            with gr.Row():
                fitness_score = gr.Number(
                    label="Fitness Score (lower = better)",
                    precision=4,
                    interactive=False
                )
                fitness_interp = gr.Textbox(
                    label="Interpretation",
                    interactive=False
                )

            comparison_details = gr.JSON(
                label="Comparison Details",
                visible=True
            )

            comparison_plot = gr.Plot(
                label="Profile Comparison (A vs B)",
            )

            compare_status = gr.Textbox(
                label="Status",
                interactive=False
            )

            # Wire up compare button
            compare_outputs = [
                fitness_score, fitness_interp, comparison_details,
                comparison_plot, compare_status,
            ]
            compare_btn.click(
                fn=compare_passages,
                inputs=[golden_input, synthetic_input, threshold_compare, window_compare],
                outputs=compare_outputs,
            )

            # Reactive slider updates (recompute from cached embeddings)
            for slider in [threshold_compare, window_compare]:
                slider.change(
                    fn=_recompute_comparison,
                    inputs=[threshold_compare, window_compare],
                    outputs=compare_outputs,
                )

    return {
        "text_input": text_input,
        "analyze_btn": analyze_btn,
        "deadpan_score": deadpan_score,
        "heller_score": heller_score,
        "trajectory_plot": trajectory_plot,
        "golden_input": golden_input,
        "synthetic_input": synthetic_input,
        "compare_btn": compare_btn,
        "fitness_score": fitness_score,
    }
