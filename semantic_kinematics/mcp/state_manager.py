"""
State manager for MCP server.

Handles embedding cache and session state across tool calls.
Uses adapter pattern to support multiple embedding backends.

Environment variables (no baked default -- Rule #14: a silently-wrong-model
run is an unacceptable failure class):
    EMBEDDING_BACKEND: "lmstudio", "nv_embed", or "sentence_transformers".
        Required (via env or set_backend()) before get_adapter()/get_embed_fn()
        is called; unresolved selection raises ValueError naming what is missing.
    EMBEDDING_SERVER_URL: API URL for the lmstudio backend.
    EMBEDDING_MODEL: Model name for API backends.
"""

import hashlib
import os
import numpy as np
from typing import Dict, Optional, Callable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from semantic_kinematics.embeddings.base import EmbeddingAdapter


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

    Backend is resolved from EMBEDDING_BACKEND (env) or set_backend() (explicit
    args); no baked default. Backend resolution is lazy -- constructing a
    StateManager never raises -- but get_adapter()/get_embed_fn() raise a clear
    ValueError if no backend was ever resolved before the adapter is needed.
    """

    _embedding_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _adapter: Optional["EmbeddingAdapter"] = None
    _backend: Optional[str] = field(default_factory=lambda: os.environ.get("EMBEDDING_BACKEND"))
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

        Raises:
            ValueError: If no backend was resolved from an explicit
                set_backend() call or the EMBEDDING_BACKEND environment
                variable. Rule #14: an unresolved backend must hard-fail
                rather than silently pick one.
        """
        if self._adapter is None:
            if not self._backend:
                raise ValueError(
                    "No embedding backend resolved: set EMBEDDING_BACKEND "
                    "(e.g. 'lmstudio', 'nv_embed', 'sentence_transformers') "
                    "or call StateManager.set_backend(...) explicitly."
                )
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
