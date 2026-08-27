"""Regression guard for issue #14: StateManager must not silently pick a
backend. Constructing a StateManager stays lazy (no env read failure at
construction time -- other tests bare-construct it and never touch the
adapter), but get_adapter()/get_embed_fn() must hard-fail with a clear
message when no backend was ever resolved from an explicit set_backend()
call or the EMBEDDING_BACKEND environment variable.
"""

from __future__ import annotations

import pytest

from semantic_kinematics.mcp.state_manager import StateManager


def test_bare_construction_does_not_raise(monkeypatch):
    """Constructing StateManager must stay safe even with no backend resolved
    -- many callers (server.py, ui/state.py, other tests) bare-construct it
    and only touch the adapter lazily, if at all."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    StateManager()  # must not raise


def test_get_adapter_without_backend_raises_clear_message(monkeypatch):
    """No EMBEDDING_BACKEND and no set_backend() call: get_adapter() must
    raise ValueError naming what is missing, not silently pick 'lmstudio'."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    manager = StateManager()

    with pytest.raises(ValueError) as excinfo:
        manager.get_adapter()

    msg = str(excinfo.value).lower()
    assert "backend" in msg
    assert "embedding_backend" in msg or "set_backend" in msg


def test_get_embed_fn_without_backend_raises(monkeypatch):
    """get_embed_fn() routes through get_adapter(); same hard-fail applies."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    manager = StateManager()

    with pytest.raises(ValueError):
        manager.get_embed_fn()


def test_env_backend_is_resolved(monkeypatch):
    """EMBEDDING_BACKEND is still honored when set (env layer of args -> env
    -> hard fail)."""
    monkeypatch.setenv("EMBEDDING_BACKEND", "lmstudio")
    monkeypatch.setenv("EMBEDDING_MODEL", "some-model")
    monkeypatch.setenv("EMBEDDING_SERVER_URL", "http://host:1234/v1")
    manager = StateManager()

    adapter = manager.get_adapter()
    assert adapter is not None


def test_set_backend_resolves_explicitly(monkeypatch):
    """set_backend() is the explicit-args layer and must work with no env
    set at all."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    manager = StateManager()
    manager.set_backend("lmstudio", model_name="some-model", base_url="http://host:1234/v1")

    adapter = manager.get_adapter()
    assert adapter is not None
