"""
Classification command module for semantic document classification.

Tools:
- classify_document: Classify document by similarity to category exemplars
"""

from typing import Any, Dict, List
from mcp.types import Tool

from semantic_kinematics.mcp.state_manager import StateManager


# Content truncation limit
MAX_CONTENT_CHARS = 2000


def get_tools() -> List[Tool]:
    """Return classification-related tool definitions."""
    return [
        Tool(
            name="classify_document",
            description=(
                "Classify document content by similarity to category exemplars. "
                "Content silently truncated to 2000 chars (head). "
                "Category exemplar embeddings cached for session."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Document content (silently truncated to 2000 chars)"
                    },
                    "categories": {
                        "type": "object",
                        "description": (
                            "Category name → exemplar text mapping. "
                            "CALLER is responsible for building exemplars "
                            "(e.g., from folder contents). semantic-kinematics "
                            "has no filesystem access."
                        ),
                        "additionalProperties": {"type": "string"}
                    },
                    "threshold": {
                        "type": "number",
                        "description": "Confidence threshold for classification",
                        "default": 0.85
                    }
                },
                "required": ["content", "categories"]
            }
        ),
    ]


def _cosine_similarity(vec_a, vec_b) -> float:
    """Calculate cosine similarity between two vectors."""
    import numpy as np
    dot = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


async def classify_document(
    manager: StateManager,
    args: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Classify document content by similarity to category exemplars.

    Uses embedding-first classification strategy:
    - Confident classifications (above threshold): procedural, fast, cheap
    - Uncertain classifications (below threshold): caller decides fallback

    Category exemplar embeddings are cached for the session, so batch
    classification of many documents reuses exemplar embeddings.
    """
    content = args.get("content", "")
    categories = args.get("categories", {})
    threshold = args.get("threshold", 0.85)

    if not content:
        return {"error": "No content provided"}

    if not categories:
        return {"error": "No categories provided"}

    # Truncate content
    content_truncated = len(content) > MAX_CONTENT_CHARS
    if content_truncated:
        content = content[:MAX_CONTENT_CHARS]

    try:
        # Get embedding function (with caching)
        embed_fn = manager.get_embed_fn()

        # Embed document content
        doc_embedding = embed_fn(content)

        # Embed each category exemplar (cached by StateManager)
        similarities = {}
        for category_name, exemplar_text in categories.items():
            if not exemplar_text:
                continue
            exemplar_embedding = embed_fn(exemplar_text)
            sim = _cosine_similarity(doc_embedding, exemplar_embedding)
            similarities[category_name] = sim

        if not similarities:
            return {"error": "No valid category exemplars provided"}

        # Find best match
        best_match = max(similarities, key=similarities.get)
        best_similarity = similarities[best_match]
        confident = best_similarity >= threshold

        return {
            "best_match": best_match,
            "similarity": round(best_similarity, 4),
            "confident": confident,
            "all_similarities": {
                k: round(v, 4)
                for k, v in sorted(similarities.items(), key=lambda x: -x[1])
            },
            "content_truncated": content_truncated,
            "model": manager.model_name,
            "threshold": threshold,
        }

    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}
