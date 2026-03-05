"""
State manager for MCP server.

Handles embedding cache and session state across tool calls.
Uses adapter pattern to support multiple embedding backends.

Environment variables:
    EMBEDDING_BACKEND: "lmstudio" (default), "nv_embed", "sentence_transformers"
    EMBEDDING_SERVER_URL: API URL for lmstudio backend (default: http://localhost:1234/v1)
    EMBEDDING_MODEL: Model name for API backends
"""

import hashlib
import os
import numpy as np
from typing import Dict, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from semantic_kinematics.embeddings.base import EmbeddingAdapter


def _default_backend() -> str:
    """Get default backend from environment, fallback to lmstudio (API-first, no torch)."""
    return os.environ.get("EMBEDDING_BACKEND", "lmstudio")


def _default_backend_kwargs() -> Dict:
    """Build default kwargs from environment variables."""
    kwargs = {}
    if url := os.environ.get("EMBEDDING_SERVER_URL"):
        kwargs["base_url"] = url
    if model := os.environ.get("EMBEDDING_MODEL"):
        kwargs["model_name"] = model
    return kwargs


@dataclass
class StateManager:
    """
    Manages state across MCP tool calls.

    Primarily handles:
    - Embedding cache to avoid re-computing embeddings
    - Embedding adapter initialization (supports multiple backends)

    Default backend: lmstudio (API-based, no torch required)
    Set EMBEDDING_BACKEND=nv_embed for local GPU inference.
    """

    _embedding_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _adapter: Optional["EmbeddingAdapter"] = None
    _backend: str = field(default_factory=_default_backend)
    _backend_kwargs: Dict = field(default_factory=_default_backend_kwargs)

    def _cache_key(self, text: str) -> str:
        """Hash-based key for cache lookup."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def get_cached_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding from cache if available."""
        key = self._cache_key(text)
        return self._embedding_cache.get(key)

    def cache_embedding(self, text: str, embedding: np.ndarray) -> None:
        """Store embedding in cache."""
        key = self._cache_key(text)
        self._embedding_cache[key] = embedding

    def clear_cache(self) -> int:
        """Clear embedding cache. Returns number of entries cleared."""
        count = len(self._embedding_cache)
        self._embedding_cache.clear()
        return count

    def get_adapter(self) -> "EmbeddingAdapter":
        """
        Get the embedding adapter, initializing if needed.

        Returns:
            Configured EmbeddingAdapter instance
        """
        if self._adapter is None:
            from semantic_kinematics.embeddings import get_adapter
            self._adapter = get_adapter(self._backend, **self._backend_kwargs)
        return self._adapter

    def get_embed_fn(self) -> Callable:
        """
        Get the embedding function with caching.

        Returns a callable that:
        1. Checks cache first
        2. Falls back to adapter.embed()
        3. Caches the result
        """
        adapter = self.get_adapter()

        def embed(text: str) -> np.ndarray:
            # Check cache first
            cached = self.get_cached_embedding(text)
            if cached is not None:
                return cached

            # Generate embedding
            embedding = adapter.embed(text)

            # Cache the result
            self.cache_embedding(text, embedding)
            return embedding

        return embed

    def set_backend(self, backend: str, **kwargs) -> None:
        """
        Switch embedding backend.

        Clears cache since different backends have different dimensions.

        Args:
            backend: "sentence_transformers" or "lmstudio"
            **kwargs: Passed to adapter constructor
        """
        self._backend = backend
        self._backend_kwargs = kwargs
        self._adapter = None  # Force re-initialization
        self._embedding_cache.clear()  # Clear cache (different dimensions)

    @property
    def model_name(self) -> str:
        """Get current model name from adapter."""
        return self.get_adapter().model_name

    @property
    def dimensions(self) -> int:
        """Get embedding dimensions from adapter."""
        return self.get_adapter().dimensions
