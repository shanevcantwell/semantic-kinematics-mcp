"""
Embedding adapters for semantic-kinematics.

Provides a unified interface for different embedding backends:
- SentenceTransformers (default): Native PyTorch, supports NV-Embed-v2
- LMStudio: OpenAI-compatible API for GGUF'd models

Usage:
    from semantic_kinematics.embeddings import get_adapter

    # Default: SentenceTransformers with NV-Embed-v2
    adapter = get_adapter()
    embedding = adapter.embed("Hello world")

    # Explicit backend selection
    adapter = get_adapter("lmstudio", model_name="nomic-embed-text-v1.5")
    adapter = get_adapter("sentence_transformers", model_path="/path/to/model")
"""

from typing import Optional

from semantic_kinematics.embeddings.base import EmbeddingAdapter


def get_adapter(
    backend: str = "nv_embed",
    **kwargs
) -> EmbeddingAdapter:
    """
    Get embedding adapter by backend name.

    Args:
        backend: Backend type:
            - "nv_embed" (default): Direct NV-Embed-v2 via transformers (fp32)
            - "sentence_transformers": SentenceTransformers wrapper
            - "lmstudio": OpenAI-compatible API for GGUF models
        **kwargs: Passed to adapter constructor

    Returns:
        Configured EmbeddingAdapter instance

    Examples:
        # Default NV-Embed-v2 in fp32
        adapter = get_adapter()

        # LM Studio with nomic
        adapter = get_adapter("lmstudio", model_name="nomic-embed-text-v1.5")

        # SentenceTransformers (may have compatibility issues)
        adapter = get_adapter("sentence_transformers", model_path="/my/model")
    """
    if backend == "nv_embed":
        from semantic_kinematics.embeddings.nv_embed_adapter import NVEmbedAdapter
        return NVEmbedAdapter(**kwargs)

    elif backend == "sentence_transformers":
        from semantic_kinematics.embeddings.sentence_transformers_adapter import (
            SentenceTransformersAdapter
        )
        return SentenceTransformersAdapter(**kwargs)

    elif backend == "lmstudio":
        from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter
        return LMStudioAdapter(**kwargs)

    else:
        raise ValueError(
            f"Unknown backend: {backend}. "
            f"Available: 'nv_embed', 'sentence_transformers', 'lmstudio'"
        )


# Convenience aliases
def get_nv_embed_adapter(**kwargs) -> EmbeddingAdapter:
    """Get adapter configured for NV-Embed-v2."""
    return get_adapter("nv_embed", **kwargs)


def get_lmstudio_adapter(**kwargs) -> EmbeddingAdapter:
    """Get adapter configured for LM Studio."""
    return get_adapter("lmstudio", **kwargs)


__all__ = [
    "EmbeddingAdapter",
    "get_adapter",
    "get_nv_embed_adapter",
    "get_lmstudio_adapter",
]
