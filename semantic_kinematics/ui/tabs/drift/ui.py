"""
UI layout for the Drift tab.

Provides single-pair and bulk semantic drift calculation.
"""

import gradio as gr

from semantic_kinematics.ui.tabs.drift.handlers import (
    calculate_drift_single,
    process_bulk_drift
)


def build_tab():
    """Build the Drift tab UI components and wire up events."""

    gr.Markdown("""
    ## Drift: Semantic Distance Calculator

    Calculate cosine distance between text pairs.

    **Scale:**
    - 0.0-0.1: Very similar (near-identical)
    - 0.1-0.3: Similar (related semantics)
    - 0.3-0.5: Moderate drift
    - 0.5-0.7: Significant drift
    - 0.7+: High drift (unrelated/opposite)
    """)

    with gr.Tabs():
        # Single mode
        with gr.Tab("Single Pair"):
            with gr.Row():
                text_a = gr.Textbox(
                    label="Text A",
                    placeholder="Enter first text...",
                    lines=4
                )
                text_b = gr.Textbox(
                    label="Text B",
                    placeholder="Enter second text...",
                    lines=4
                )

            single_btn = gr.Button("Calculate Drift", variant="primary")

            with gr.Row():
                drift_value = gr.Number(
                    label="Drift",
                    precision=4,
                    interactive=False
                )
                interpretation = gr.Textbox(
                    label="Interpretation",
                    interactive=False
                )
                single_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    visible=False
                )

            # Wire up single mode
            single_btn.click(
                fn=calculate_drift_single,
                inputs=[text_a, text_b],
                outputs=[drift_value, interpretation, single_status]
            )

        # Bulk mode
        with gr.Tab("Bulk (JSONL)"):
            gr.Markdown("""
            Upload a JSONL file with one pair per line:
            ```
            {"text_a": "First text A", "text_b": "First text B"}
            {"text_a": "Second text A", "text_b": "Second text B"}
            ```
            """)

            bulk_file = gr.File(
                label="Upload JSONL file",
                file_types=[".jsonl", ".json"],
                type="filepath"
            )

            bulk_btn = gr.Button("Process Batch", variant="primary")

            bulk_status = gr.Textbox(
                label="Status",
                interactive=False
            )

            results_table = gr.Dataframe(
                headers=["Line", "Text A", "Text B", "Drift", "Interpretation"],
                datatype=["number", "str", "str", "number", "str"],
                label="Results",
                wrap=True
            )

            download_csv = gr.File(
                label="Download results CSV",
                interactive=False
            )

            # Wire up bulk mode
            bulk_btn.click(
                fn=process_bulk_drift,
                inputs=[bulk_file],
                outputs=[results_table, download_csv, bulk_status]
            )

    return {
        "text_a": text_a,
        "text_b": text_b,
        "single_btn": single_btn,
        "drift_value": drift_value,
        "interpretation": interpretation,
        "bulk_file": bulk_file,
        "bulk_btn": bulk_btn,
        "results_table": results_table,
        "download_csv": download_csv
    }
