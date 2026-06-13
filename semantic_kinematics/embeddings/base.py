"""
Base class for embedding adapters.

Defines the interface for embedding backends (LM Studio, SentenceTransformers, etc.)
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingAdapter(ABC):
    """
    Abstract base for embedding backends.

    Implementations provide consistent interface for different embedding sources:
    - LMStudioAdapter: OpenAI-compatible API for GGUF'd models
    - SentenceTransformersAdapter: Native PyTorch for non-GGUF models (NV-Embed-v2)
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Embedding vector dimensions."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        pass

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        Default implementation is sequential. Subclasses can override
        for optimized batch processing.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (len(texts), dimensions)
        """
        return np.array([self.embed(t) for t in texts])

    def count_tokens(self, text: str) -> int:
        """
        Count the tokens the backend's tokenizer produces for ``text``.

        This is part of the adapter contract because only the backend knows
        its own tokenizer; a chars-per-token heuristic undershoots badly on
        dense text (code/JSON) and lets oversized inputs slip past the
        sub-chunk splitter into server-side context overflow.

        Subclasses that can reach a real tokenizer (the server's ``/tokenize``
        endpoint, a resident HF tokenizer, etc.) must override this and return
        an exact count. Backends with no reachable tokenizer leave the default,
        which raises so callers fail loudly rather than splitting on a fiction.

        Args:
            text: Input text to tokenize.

        Returns:
            Exact token count.

        Raises:
            NotImplementedError: if the backend exposes no tokenizer.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement count_tokens; "
            "token-aware splitting is unavailable for this backend."
        )

    def unload(self) -> None:
        """Unload model from memory. No-op for stateless backends."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Whether the model is currently loaded. Always True for stateless backends."""
        return True

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec_a: First embedding vector
            vec_b: Second embedding vector

        Returns:
            Similarity score (0.0 to 1.0 for normalized vectors)
        """
        dot = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def cosine_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculate cosine distance between two vectors.

        Args:
            vec_a: First embedding vector
            vec_b: Second embedding vector

        Returns:
            Distance score (0.0 = identical, 1.0 = orthogonal, 2.0 = opposite)
        """
        return 1.0 - self.cosine_similarity(vec_a, vec_b)
