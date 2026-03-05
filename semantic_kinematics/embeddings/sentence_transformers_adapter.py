"""
SentenceTransformers embedding adapter.

Native PyTorch backend for models that can't be GGUF'd (e.g., NV-Embed-v2).
"""

from typing import List, Optional

import numpy as np

from semantic_kinematics.embeddings.base import EmbeddingAdapter


class SentenceTransformersAdapter(EmbeddingAdapter):
    """
    Native PyTorch adapter for SentenceTransformers models.

    Used for models that require trust_remote_code=True or can't be
    converted to GGUF format (e.g., NV-Embed-v2 with 4096 dimensions).

    Requires GPU for reasonable performance with large models.
    """

    # Default path to NV-Embed-v2 model
    DEFAULT_MODEL_PATH = "/home/shane/github/NV-Embed-v2"

    def __init__(
        self,
        model_path: Optional[str] = None,
        normalize: bool = True,
        device: Optional[str] = None,
        use_fp16: bool = False,
    ):
        """
        Initialize SentenceTransformers adapter.

        Args:
            model_path: Path to model directory (default: NV-Embed-v2)
            normalize: Whether to L2-normalize embeddings
            device: Device to use ('cuda', 'cpu', or None for auto)
            use_fp16: Use float16 precision to reduce memory (default: False; set True for 24GB GPUs)
        """
        self._model_path = model_path or self.DEFAULT_MODEL_PATH
        self._normalize = normalize
        self._device = device
        self._use_fp16 = use_fp16
        self._model = None
        self._dimensions: Optional[int] = None

    def _get_model(self):
        """Lazy initialization of SentenceTransformer model."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            kwargs = {"trust_remote_code": True}
            if self._device:
                kwargs["device"] = self._device

            # HuggingFace NV-Embed-v2 is already fp16 on disk.
            # Pass model_kwargs to prevent upcasting to fp32 during load.
            if self._use_fp16:
                kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

            self._model = SentenceTransformer(self._model_path, **kwargs)

        return self._model

    @property
    def model_name(self) -> str:
        return f"SentenceTransformers:{self._model_path}"

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            # For NV-Embed-v2, we know it's 4096
            if "NV-Embed-v2" in self._model_path:
                self._dimensions = 4096
            else:
                # Probe with dummy embed
                test = self.embed("test")
                self._dimensions = len(test)
        return self._dimensions

    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding via SentenceTransformers.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as numpy array
        """
        model = self._get_model()
        embedding = model.encode(
            [text],
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return embedding[0]

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for multiple texts with GPU batching.

        SentenceTransformers handles batching efficiently on GPU.

        Args:
            texts: List of input texts

        Returns:
            Array of shape (len(texts), dimensions)
        """
        if not texts:
            return np.array([])

        model = self._get_model()
        return model.encode(
            texts,
            normalize_embeddings=self._normalize,
            batch_size=32,
            show_progress_bar=len(texts) > 10,
        )
