"""Regression guard: embed_corpus must route adapter kwargs by backend.

The CLI previously called ``get_adapter(backend, model_name=..., base_url=...)``
unconditionally. Those kwargs are lmstudio-only; the in-process backends
(``nv_embed`` / ``sentence_transformers``) take ``model_path`` and raise
``TypeError`` on the leaked kwargs — so ``--backend nv_embed`` crashed before
loading anything. ``_make_adapter`` routes the kwargs per backend; these tests
pin that routing without constructing any real adapter.
"""

from __future__ import annotations

import pytest

import scripts.embed_corpus as ec


def _capturing_get_adapter(captured):
    def _fake(backend, **kwargs):
        captured["backend"] = backend
        captured["kwargs"] = kwargs
        return object()

    return _fake


def test_lmstudio_backend_gets_network_kwargs(monkeypatch):
    captured = {}
    monkeypatch.setattr(ec, "get_adapter", _capturing_get_adapter(captured))

    ec._make_adapter("lmstudio", "some-model", "http://host:1234/v1")

    assert captured["backend"] == "lmstudio"
    assert captured["kwargs"] == {
        "model_name": "some-model",
        "base_url": "http://host:1234/v1",
    }


def test_in_process_backend_drops_network_kwargs(monkeypatch):
    """sentence_transformers (path-based) must NOT receive model_name/base_url —
    forwarding them is the TypeError regression."""
    captured = {}
    monkeypatch.setattr(ec, "get_adapter", _capturing_get_adapter(captured))

    ec._make_adapter("sentence_transformers", "embeddinggemma-300M-F32", "http://host:8082/v1")

    assert captured["backend"] == "sentence_transformers"
    assert captured["kwargs"] == {}


def test_nv_embed_backend_is_resident_and_drops_network_kwargs(monkeypatch):
    """nv_embed must drop the lmstudio-only kwargs AND request a resident model
    (unload_after_use=False) so a corpus run pays one model load, not one per
    request-group."""
    captured = {}
    monkeypatch.setattr(ec, "get_adapter", _capturing_get_adapter(captured))

    ec._make_adapter("nv_embed", "embeddinggemma-300M-F32", "http://host:8082/v1")

    assert captured["backend"] == "nv_embed"
    assert captured["kwargs"] == {"unload_after_use": False}
