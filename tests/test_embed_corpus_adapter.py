"""Regression guard: embed_corpus must route adapter kwargs by backend.

The CLI previously called ``get_adapter(backend, model_name=..., base_url=...)``
unconditionally. Those kwargs are lmstudio-only; the in-process backends
(``nv_embed`` / ``sentence_transformers``) take ``model_path`` and raise
``TypeError`` on the leaked kwargs — so ``--backend nv_embed`` crashed before
loading anything. ``_make_adapter`` routes the kwargs per backend; these tests
pin that routing without constructing any real adapter.

Also guards issue #14: ``--model``/``--base-url`` must not silently default to
a specific model/endpoint (previously ``embeddinggemma-300M-F32`` /
``http://localhost:8082/v1``) when ``--backend lmstudio`` is used -- a
silently-wrong-model run is an unacceptable failure class.
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


def test_main_requires_model_for_lmstudio_backend(tmp_path, monkeypatch):
    """No --model, no EMBEDDING_MODEL, --backend lmstudio: must exit loudly
    rather than silently embedding with a baked-in model id."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_SERVER_URL", raising=False)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "hi", "chunk_id": "1"}\n')
    checkpoint = tmp_path / "checkpoint.jsonl"

    with pytest.raises(SystemExit):
        ec.main([
            str(corpus), "--checkpoint", str(checkpoint),
            "--backend", "lmstudio", "--base-url", "http://host:8082/v1",
        ])


def test_main_requires_base_url_for_lmstudio_backend(tmp_path, monkeypatch):
    """No --base-url, no EMBEDDING_SERVER_URL, --backend lmstudio: must exit
    loudly rather than silently pointing at localhost:8082."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_SERVER_URL", raising=False)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"text": "hi", "chunk_id": "1"}\n')
    checkpoint = tmp_path / "checkpoint.jsonl"

    with pytest.raises(SystemExit):
        ec.main([
            str(corpus), "--checkpoint", str(checkpoint),
            "--backend", "lmstudio", "--model", "some-model",
        ])


def test_main_lmstudio_model_env_fallback(monkeypatch):
    """EMBEDDING_MODEL/EMBEDDING_SERVER_URL env vars satisfy the requirement
    when --model/--base-url are omitted (args -> env -> hard fail)."""
    monkeypatch.setenv("EMBEDDING_MODEL", "some-model")
    monkeypatch.setenv("EMBEDDING_SERVER_URL", "http://host:8082/v1")
    import importlib
    importlib.reload(ec)
    try:
        captured = {}

        def _fake_make_adapter(backend, model, base_url):
            captured["model"] = model
            captured["base_url"] = base_url
            raise SystemExit(0)  # short-circuit before any real embedding work

        monkeypatch.setattr(ec, "_make_adapter", _fake_make_adapter)
        monkeypatch.setattr(ec, "_read_items", lambda *a, **k: [("hi", "1")])

        with pytest.raises(SystemExit):
            ec.main(["corpus.jsonl", "--checkpoint", "out.jsonl", "--backend", "lmstudio"])

        assert captured["model"] == "some-model"
        assert captured["base_url"] == "http://host:8082/v1"
    finally:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_SERVER_URL", raising=False)
        importlib.reload(ec)
