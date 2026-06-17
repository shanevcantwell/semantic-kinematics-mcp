"""
LM Studio embedding adapter.

Uses OpenAI-compatible API to connect to LM Studio for GGUF'd embedding models.
"""

from typing import List, Optional

import numpy as np
import requests

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
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LM Studio adapter.

        Args:
            model_name: Model identifier in LM Studio (required; no default)
            base_url: LM Studio API endpoint (required; no default)

        Raises:
            ValueError: If model_name or base_url is missing. Rule #14 forbids a
                baked-in default — an implicit nomic model / LM-Studio endpoint
                is a silently-wrong-model failure class.
        """
        if not model_name:
            raise ValueError(
                "no embedding model specified; pass 'model_name'"
            )
        if not base_url:
            raise ValueError(
                "no embedding endpoint specified; pass 'base_url'"
            )
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

    def _tokenize_url(self) -> str:
        """Derive the server-root ``/tokenize`` URL from the ``/v1`` base_url.

        llama.cpp exposes ``/tokenize`` at the server root, not under ``/v1``.
        """
        # Assumes ``/v1`` is the terminal path component of base_url (the
        # OpenAI-compatible convention); only that trailing segment is stripped.
        root = self._base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root.rstrip('/')}/tokenize"

    def _native_embeddings_url(self) -> str:
        """Derive the server-root ``/embeddings`` URL from the ``/v1`` base_url.

        llama.cpp's native ``/embeddings`` endpoint (server root, not ``/v1``)
        returns **per-token** vectors when the server runs ``--pooling none`` —
        unlike the OpenAI-compatible ``/v1/embeddings`` path, which always pools.
        """
        root = self._base_url.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root.rstrip('/')}/embeddings"

    def tokenize_pieces(self, text: str) -> List[dict]:
        """Return ``[{"id": int, "piece": str}, ...]`` for ``text``.

        Uses llama.cpp ``/tokenize`` with ``with_pieces=true``. The pieces are
        the **content** tokens (no BOS/EOS) and concatenate back to ``text``
        (leading spaces are part of the piece, Gemma convention), so per-token
        character offsets can be reconstructed exactly for span localization.
        """
        response = requests.post(
            self._tokenize_url(), json={"content": text, "with_pieces": True}
        )
        response.raise_for_status()
        payload = response.json()
        tokens = payload.get("tokens") if isinstance(payload, dict) else payload
        if not isinstance(tokens, list) or (
            tokens and not isinstance(tokens[0], dict)
        ):
            body = repr(payload)
            if len(body) > 500:
                body = body[:500] + "...(truncated)"
            raise ValueError(
                f"/tokenize with_pieces returned unexpected shape; body={body}"
            )
        return tokens

    def embed_tokens(self, text: str) -> np.ndarray:
        """Return **per-token** embeddings for ``text``: shape ``(n_rows, dim)``.

        POSTs ``{"content": text}`` to the native ``/embeddings`` endpoint, which
        (server in ``--pooling none``) returns ``[{"index": 0, "embedding":
        [[...], ...]}]`` — a 2-D matrix per item. Rows are ``[BOS] + content +
        [EOS]``; callers align ``content`` against :meth:`tokenize_pieces` and
        trim the two specials.

        Raises:
            ValueError: if the response is pooled (1 row) — the server is not in
                ``--pooling none`` — or otherwise not the expected 2-D shape.
        """
        response = requests.post(
            self._native_embeddings_url(), json={"content": text}
        )
        response.raise_for_status()
        payload = response.json()
        try:
            matrix = np.asarray(payload[0]["embedding"], dtype=float)
        except (KeyError, IndexError, TypeError) as exc:
            body = repr(payload)
            if len(body) > 500:
                body = body[:500] + "...(truncated)"
            raise ValueError(
                f"native /embeddings unexpected shape ({exc}); body={body}"
            ) from exc
        if matrix.ndim != 2:
            raise ValueError(
                "native /embeddings did not return per-token vectors "
                f"(got shape {matrix.shape}); is the server in --pooling none?"
            )
        return matrix

    def count_tokens(self, text: str) -> int:
        """
        Return the exact token count for ``text`` via the server's tokenizer.

        POSTs to llama.cpp's ``/tokenize`` endpoint (server root, not ``/v1``),
        which returns ``{"tokens": [...]}``; the count is the length of that
        list. This is the real tokenizer, so the split decision no longer rests
        on a chars-per-token fiction that undershoots on dense code/JSON.

        Args:
            text: Input text to tokenize.

        Returns:
            Exact token count.
        """
        response = requests.post(self._tokenize_url(), json={"content": text})
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict) or "tokens" not in payload:
            body = repr(payload)
            if len(body) > 500:
                body = body[:500] + "...(truncated)"
            raise ValueError(
                f"/tokenize returned 200 but no 'tokens' field; body={body}"
            )
        return len(payload["tokens"])

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
