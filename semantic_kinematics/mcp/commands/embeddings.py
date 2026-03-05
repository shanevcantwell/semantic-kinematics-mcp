"""
Embeddings command module for direct embedding operations.

Tools:
- embed_text: Get embedding vector for text
- calculate_drift: Calculate semantic drift between two texts
"""

from typing import Any, Dict, List
from mcp.types import Tool

from semantic_kinematics.mcp.state_manager import StateManager


def get_tools() -> List[Tool]:
    """Return embedding-related tool definitions."""
    return [
        Tool(
            name="embed_text",
            description=(
                "Get embedding vector for text. "
                "Default: returns first 10 dims for readability. "
                "With full_vector=true: ~50KB JSON (4096 dims × float32 × JSON overhead)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to embed"
                    },
                    "full_vector": {
                        "type": "boolean",
                        "description": "Return full embedding vector (can be large)",
                        "default": False
                    },
                    "model": {
                        "type": "string",
                        "description": "Embedding model to use",
                        "default": "nomic-embed-text-v1.5"
                    }
                },
                "required": ["text"]
            }
        ),
        Tool(
            name="calculate_drift",
            description="Calculate semantic drift (cosine distance) between two texts",
            inputSchema={
                "type": "object",
                "properties": {
                    "text_a": {
                        "type": "string",
                        "description": "First text"
                    },
                    "text_b": {
                        "type": "string",
                        "description": "Second text"
                    }
                },
                "required": ["text_a", "text_b"]
            }
        ),
    ]


async def embed_text(
    manager: StateManager,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get embedding vector for text.

    Returns truncated vector by default for readability.
    """
    text = args.get("text", "")
    full_vector = args.get("full_vector", False)
    model = args.get("model", "nomic-embed-text-v1.5")

    if not text:
        return {"error": "No text provided"}

    try:
        # Note: model parameter is informational only
        # All tools use the configured backend (default: NV-Embed-v2)
        embed_fn = manager.get_embed_fn()
        embedding = embed_fn(text)

        result = {
            "text_preview": text[:100] + "..." if len(text) > 100 else text,
            "model": model,
            "dimensions": len(embedding),
        }

        if full_vector:
            result["embedding"] = embedding.tolist()
        else:
            result["embedding_preview"] = embedding[:10].tolist()
            result["note"] = "Use full_vector=true for complete embedding"

        return result

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Embedding failed: {str(e)}"}


async def calculate_drift(
    manager: StateManager,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Calculate semantic drift (cosine distance) between two texts.

    Drift = 0.0 means identical
    Drift = 1.0 means orthogonal
    Drift = 2.0 means opposite
    """
    text_a = args.get("text_a", "")
    text_b = args.get("text_b", "")

    if not text_a or not text_b:
        return {"error": "Both text_a and text_b are required"}

    try:
        from semantic_kinematics.prompt_geometry.metrics import cosine_distance

        embed_fn = manager.get_embed_fn()

        vec_a = embed_fn(text_a)
        vec_b = embed_fn(text_b)

        drift = cosine_distance(vec_a, vec_b)

        # Interpret the drift
        if drift < 0.1:
            interpretation = "Very similar (near-identical semantics)"
        elif drift < 0.3:
            interpretation = "Similar (related semantics)"
        elif drift < 0.5:
            interpretation = "Moderate drift (some semantic divergence)"
        elif drift < 0.7:
            interpretation = "Significant drift (different semantics)"
        else:
            interpretation = "High drift (unrelated or opposite semantics)"

        return {
            "drift": round(drift, 4),
            "interpretation": interpretation,
            "text_a_preview": text_a[:80] + "..." if len(text_a) > 80 else text_a,
            "text_b_preview": text_b[:80] + "..." if len(text_b) > 80 else text_b,
        }

    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Drift calculation failed: {str(e)}"}
