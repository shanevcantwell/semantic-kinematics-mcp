"""
Event handlers for the Drift tab.

Calculates semantic distance (cosine drift) between texts.
Supports single pair and bulk (JSONL) processing.
"""

import csv
import json
import tempfile
from pathlib import Path
from typing import Any

from semantic_kinematics.mcp.commands.embeddings import calculate_drift
from semantic_kinematics.ui.state import state_manager, drift_session


async def calculate_drift_single(
    text_a: str,
    text_b: str
) -> tuple[float | None, str, str]:
    """
    Calculate drift between two texts.

    Returns:
        (drift_value, interpretation, status_message)
    """
    if not text_a or not text_b:
        return None, "", "Both texts are required"

    result = await calculate_drift(state_manager, {
        "text_a": text_a,
        "text_b": text_b
    })

    if "error" in result:
        return None, "", f"Error: {result['error']}"

    # Update session history
    drift_session.history.append({
        "text_a_preview": result["text_a_preview"],
        "text_b_preview": result["text_b_preview"],
        "drift": result["drift"],
        "interpretation": result["interpretation"]
    })

    return result["drift"], result["interpretation"], "Calculated successfully"


async def process_bulk_drift(
    file: Any | None
) -> tuple[list[list], str | None, str]:
    """
    Process bulk drift calculations from JSONL file.

    Expects JSONL with lines: {"text_a": "...", "text_b": "..."}

    Returns:
        (results_data_for_dataframe, download_path, status_message)
    """
    if not file:
        return [], None, "No file uploaded"

    file_path = file.name if hasattr(file, 'name') else str(file)

    try:
        # Parse JSONL
        pairs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    text_a = item.get("text_a", "")
                    text_b = item.get("text_b", "")
                    if text_a and text_b:
                        pairs.append((text_a, text_b, line_num))
                    else:
                        pairs.append((None, None, line_num))  # Mark invalid
                except json.JSONDecodeError:
                    pairs.append((None, None, line_num))

        if not pairs:
            return [], None, "No valid pairs found in file"

        # Process all pairs
        results = []
        for text_a, text_b, line_num in pairs:
            if text_a is None:
                results.append({
                    "line": line_num,
                    "text_a_preview": "(invalid)",
                    "text_b_preview": "(invalid)",
                    "drift": None,
                    "interpretation": "Invalid JSON or missing fields"
                })
                continue

            result = await calculate_drift(state_manager, {
                "text_a": text_a,
                "text_b": text_b
            })

            if "error" in result:
                results.append({
                    "line": line_num,
                    "text_a_preview": text_a[:50] + "..." if len(text_a) > 50 else text_a,
                    "text_b_preview": text_b[:50] + "..." if len(text_b) > 50 else text_b,
                    "drift": None,
                    "interpretation": f"Error: {result['error']}"
                })
            else:
                results.append({
                    "line": line_num,
                    "text_a_preview": result["text_a_preview"],
                    "text_b_preview": result["text_b_preview"],
                    "drift": result["drift"],
                    "interpretation": result["interpretation"]
                })

        # Create download file
        download_path = _create_results_csv(results)

        # Convert to dataframe format
        df_data = [
            [r["line"], r["text_a_preview"], r["text_b_preview"], r["drift"], r["interpretation"]]
            for r in results
        ]

        valid_count = sum(1 for r in results if r["drift"] is not None)
        status = f"Processed {len(results)} pairs ({valid_count} valid)"

        return df_data, download_path, status

    except Exception as e:
        return [], None, f"Error processing file: {str(e)}"


def _create_results_csv(results: list[dict]) -> str:
    """Create CSV file with results and return path."""
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='_drift_results.csv',
        delete=False,
        encoding='utf-8',
        newline=''
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["line", "text_a_preview", "text_b_preview", "drift", "interpretation"]
        )
        writer.writeheader()
        writer.writerows(results)
        return f.name
