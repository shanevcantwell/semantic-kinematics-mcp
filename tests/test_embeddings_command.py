"""Unit tests for the embeddings MCP command handlers.

Focus: Rule #14 — no baked-in model default. Invoking embed_text without a
'model' arg must NOT silently embed with nomic; it must return a structured
error naming what is missing. The handler returns this error before ever
touching the StateManager, so a sentinel manager (whose use would be a bug) is
sufficient.
"""

import asyncio

import pytest

from semantic_kinematics.mcp.commands import embeddings


class _ExplodingManager:
    """If the handler reaches the embedding backend despite a missing model,
    that is the silently-wrong-model bug; calling this manager makes it loud."""

    def get_embed_fn(self):
        raise AssertionError(
            "embed_fn must NOT be called when 'model' is missing "
            "(would silently embed with the backend default)"
        )


def _run(coro):
    return asyncio.run(coro)


def test_embed_text_without_model_returns_structured_error():
    result = _run(
        embeddings.embed_text(_ExplodingManager(), {"text": "hello world"})
    )
    assert "error" in result
    assert "model" in result["error"].lower()
    # It must not have produced an embedding result.
    assert "embedding" not in result
    assert "embedding_preview" not in result
    assert "dimensions" not in result


def test_embed_text_with_empty_model_returns_structured_error():
    result = _run(
        embeddings.embed_text(
            _ExplodingManager(), {"text": "hello", "model": ""}
        )
    )
    assert "error" in result
    assert "model" in result["error"].lower()


def test_embed_text_schema_requires_model_and_has_no_default():
    tools = {t.name: t for t in embeddings.get_tools()}
    schema = tools["embed_text"].inputSchema
    assert "model" in schema["required"]
    assert "default" not in schema["properties"]["model"]


def test_embed_text_with_model_reaches_backend():
    """With a model supplied, the handler proceeds to the backend. We confirm
    the model gate is passed by observing the manager is consulted."""

    class _CountingManager:
        def __init__(self):
            self.called = False

        def get_embed_fn(self):
            self.called = True

            import numpy as np

            return lambda text: np.arange(16, dtype=float)

    mgr = _CountingManager()
    result = _run(
        embeddings.embed_text(mgr, {"text": "hello", "model": "some-model"})
    )
    assert mgr.called is True
    assert result.get("model") == "some-model"
    assert result["dimensions"] == 16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
