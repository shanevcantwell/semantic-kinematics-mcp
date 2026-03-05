"""
LM Studio embedding adapter.

Uses OpenAI-compatible API to connect to LM Studio for GGUF'd embedding models.
"""

from typing import List, Optional

import numpy as np

from semantic_kinematics.embeddings.base import EmbeddingAdapter


class LMStudioAdapter(EmbeddingAdapter):
    """
    OpenAI-compatible API adapter for GGUF'd models via LM Studio.

    Supports any embedding model loaded in LM Studio that exposes
    the /v1/embeddings endpoint.

    Example models:
    - nomic-embed-text-v1.5 (768 dimensions)
    - text-embedding-3-small (1536 dimensions)
    """

    def __init__(
        self,
        model_name: str = "text-embedding-nomic-embed-text-v1.5",
        base_url: str = "http://localhost:1234/v1",
    ):
        """
        Initialize LM Studio adapter.

        Args:
            model_name: Model identifier in LM Studio
            base_url: LM Studio API endpoint
        """
        self._model_name_str = model_name
        self._base_url = base_url
        self._dimensions: Optional[int] = None
        self._client = None

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self._base_url,
                api_key="not-needed"
            )
        return self._client

    @property
    def model_name(self) -> str:
        return f"LMStudio:{self._model_name_str}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # Probe with dummy embed to discover dimensions
            test = self.embed("test")
            self._dimensions = len(test)
        return self._dimensions

    def unload(self) -> None:
        """Clear HTTP client (frees connection pool)."""
        self._client = None

    @property
    def is_loaded(self) -> bool:
        """Client is considered loaded once initialized."""
        return self._client is not None

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding via LM Studio API.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name_str,
            input=text
        )
        return np.array(response.data[0].embedding)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts.

        LM Studio supports batch embedding in a single API call.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        client = self._get_client()
        response = client.embeddings.create(
            model=self._model_name_str,
            input=texts
        )

        # Response data is in same order as input
        embeddings = [np.array(item.embedding) for item in response.data]
        return np.array(embeddings)
