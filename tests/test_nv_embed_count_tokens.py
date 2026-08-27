"""Regression guard: NVEmbedAdapter.count_tokens (issue #20 / bulk-embed).

BulkEmbedder requires ``count_tokens`` with no fallback. NVEmbedAdapter
previously inherited the base ABC's NotImplementedError, so every item failed
token-prep before the model loaded and ``--backend nv_embed`` produced only
``_failed`` placeholders. These tests pin that count_tokens delegates to the
tokenizer and that it loads the tokenizer without the full model.
"""

from __future__ import annotations

import pytest

# nv_embed_adapter imports torch at module scope (needed at runtime for the
# GPU backend); torch is an optional `gpu` extra (pyproject.toml), not part
# of the base dev install, so collecting this module without it must skip
# with a clear reason rather than fail the whole suite (issue #62).
torch = pytest.importorskip("torch", reason="torch not installed (optional `gpu` extra)")

from semantic_kinematics.embeddings.nv_embed_adapter import NVEmbedAdapter


class _FakeTokenizer:
    def encode(self, text):
        # token-per-whitespace-word, deterministic and dependency-free
        return list(range(len(text.split())))


def test_count_tokens_delegates_to_tokenizer(monkeypatch):
    adapter = NVEmbedAdapter()
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda: _FakeTokenizer())

    assert adapter.count_tokens("one two three four") == 4
    assert adapter.count_tokens("") == 0


def test_count_tokens_does_not_load_the_model(monkeypatch):
    """count_tokens must not trigger the ~15GB model load."""
    adapter = NVEmbedAdapter()
    monkeypatch.setattr(adapter, "_load_tokenizer", lambda: _FakeTokenizer())

    def _boom(*a, **k):  # pragma: no cover - must never be called
        raise AssertionError("count_tokens must not load the model")

    monkeypatch.setattr(adapter, "_load_model", _boom)

    adapter.count_tokens("some text here")
    assert adapter.is_loaded is False


def test_load_tokenizer_caches_and_is_load_model_safe(monkeypatch):
    """_load_tokenizer must instantiate the tokenizer once and cache it, so a
    later _load_model finds it non-None and does not re-instantiate (the guard
    the resident/count_tokens paths both rely on)."""
    import transformers

    calls = []

    def _fake_from_pretrained(path, *a, **k):
        calls.append(path)
        return _FakeTokenizer()

    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", _fake_from_pretrained
    )

    adapter = NVEmbedAdapter()
    first = adapter._load_tokenizer()
    second = adapter._load_tokenizer()

    assert first is second  # cached, same instance
    assert len(calls) == 1  # from_pretrained called exactly once

    # A pre-set tokenizer (e.g. populated by count_tokens) is reused, never
    # reloaded -- this is the `if self._tokenizer is None` guard _load_model
    # shares, so a count_tokens-before-embed call costs no extra tokenizer load.
    sentinel = object()
    adapter._tokenizer = sentinel
    assert adapter._load_tokenizer() is sentinel
    assert len(calls) == 1  # still only the one call


def test_default_model_path_honors_nv_embed_model_path_env(monkeypatch):
    """The var named for this backend must actually control it (the var named
    NV_EMBED_MODEL_PATH previously only affected the sentence_transformers
    backend; nv_embed hardcoded its path)."""
    import importlib

    from semantic_kinematics.embeddings import nv_embed_adapter as nva

    monkeypatch.delenv("NV_EMBED_MODEL_PATH", raising=False)
    importlib.reload(nva)
    assert nva.NVEmbedAdapter.DEFAULT_MODEL_PATH == "nvidia/NV-Embed-v2"

    monkeypatch.setenv("NV_EMBED_MODEL_PATH", "/srv/models/NV-Embed-v2")
    importlib.reload(nva)
    try:
        assert nva.NVEmbedAdapter.DEFAULT_MODEL_PATH == "/srv/models/NV-Embed-v2"
    finally:
        monkeypatch.delenv("NV_EMBED_MODEL_PATH", raising=False)
        importlib.reload(nva)
