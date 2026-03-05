"""
Main Gradio application for semantic-kinematics tools.

Usage:
    python -m semantic_kinematics
    semantic-kinematics-ui
"""

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from semantic_kinematics.ui.tabs.drift import ui as drift_ui
from semantic_kinematics.ui.tabs.trajectory import ui as trajectory_ui


def create_app() -> gr.Blocks:
    with gr.Blocks(
        title="Semantic Kinematics",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown("""
        # Semantic Kinematics

        Embedding space analysis for text.

        | Tab | Purpose |
        |-----|---------|
        | **Drift** | Cosine distance between text pairs |
        | **Trajectory** | Velocity, acceleration, and curvature of prose |
        """)

        with gr.Tabs():
            with gr.Tab("Drift"):
                drift_ui.build_tab()

            with gr.Tab("Trajectory"):
                trajectory_ui.build_tab()

    return app


def main():
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
