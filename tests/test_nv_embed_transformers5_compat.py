"""Regression guard: transformers>=5.x compat for NVEmbedAdapter (issue #55).

NV-Embed-v2's custom modeling code (``NVEmbedModel``) calls
``super().__init__(config)`` but never ``self.post_init()``. In
transformers>=5.0, ``PreTrainedModel.post_init()`` is what sets
``self.all_tied_weights_keys`` on the instance, and
``AutoModel.from_pretrained()`` reads that attribute back with plain
attribute access (no ``getattr`` default) while finalizing weight loading --
partway through the same call, before it can return a model instance to
patch. Without the fix, loading NV-Embed-v2 on transformers>=5.x raises::

    AttributeError: NVEmbedModel object has no attribute all_tied_weights_keys

These tests are import-guarded (``pytest.importorskip``) so they skip
cleanly rather than erroring collection when torch/transformers are absent
from the dev environment (issue #62), matching how the rest of the nv_embed
adapter's torch dependency is treated.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from transformers.modeling_utils import PreTrainedModel

from semantic_kinematics.embeddings.nv_embed_adapter import (
    NVEmbedAdapter,
    _ensure_all_tied_weights_keys_default,
)


@pytest.fixture(autouse=True)
def _restore_pretrained_model_attr():
    """Undo any class-level mutation so tests don't leak state into each other
    or into other test modules that import transformers."""
    had_attr = "all_tied_weights_keys" in PreTrainedModel.__dict__
    original = PreTrainedModel.__dict__.get("all_tied_weights_keys")
    try:
        yield
    finally:
        if had_attr:
            PreTrainedModel.all_tied_weights_keys = original
        elif "all_tied_weights_keys" in PreTrainedModel.__dict__:
            del PreTrainedModel.all_tied_weights_keys


def test_ensure_all_tied_weights_keys_default_sets_missing_attribute():
    """Reproduces the transformers>=5.x gap directly: when PreTrainedModel
    (the base every trust_remote_code custom model class inherits from) has
    no all_tied_weights_keys, the helper must add a safe fallback so plain
    attribute access on any subclass instance succeeds instead of raising
    AttributeError."""
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        del PreTrainedModel.all_tied_weights_keys
    assert not hasattr(PreTrainedModel, "all_tied_weights_keys")

    _ensure_all_tied_weights_keys_default()

    assert hasattr(PreTrainedModel, "all_tied_weights_keys")
    assert PreTrainedModel.all_tied_weights_keys == {}


def test_ensure_all_tied_weights_keys_default_is_noop_when_present():
    """Must not clobber an existing value -- e.g. on transformers versions
    that already define this attribute, or if a prior call already set it."""
    sentinel = {"already": "set"}
    PreTrainedModel.all_tied_weights_keys = sentinel

    _ensure_all_tied_weights_keys_default()

    assert PreTrainedModel.all_tied_weights_keys is sentinel


def test_custom_model_missing_post_init_no_longer_raises_attributeerror():
    """End-to-end reproduction of the NVEmbedModel bug shape: a custom
    PreTrainedModel subclass whose __init__ calls super().__init__(config)
    but never self.post_init() (exactly what NV-Embed-v2's modeling_nvembed.py
    does) must not raise AttributeError on all_tied_weights_keys access after
    the adapter's fix has run, regardless of what transformers version set on
    the class before."""
    if hasattr(PreTrainedModel, "all_tied_weights_keys"):
        del PreTrainedModel.all_tied_weights_keys

    class _Config(transformers.PretrainedConfig):
        model_type = "fake_nvembed_regression"

    class _ModelMissingPostInit(PreTrainedModel):
        config_class = _Config

        def __init__(self, config):
            super().__init__(config)
            # Deliberately never calls self.post_init(), mirroring
            # NVEmbedModel's __init__ in modeling_nvembed.py.

    model = _ModelMissingPostInit(_Config())
    assert not hasattr(model, "all_tied_weights_keys")  # confirms the gap exists pre-fix

    _ensure_all_tied_weights_keys_default()

    # Now falls through to the class-level default via normal attribute lookup.
    assert model.all_tied_weights_keys == {}


def test_load_model_calls_the_compat_shim_before_from_pretrained(monkeypatch):
    """Pins that _load_model() actually wires the fix in: the shim must run
    before AutoModel.from_pretrained() is called, since the real failure (see
    issue #55) happens inside that call, not after it returns."""
    calls = []

    def _fake_ensure():
        calls.append("ensure")

    def _fake_from_pretrained(*args, **kwargs):
        calls.append("from_pretrained")
        raise RuntimeError("stop before actually loading a 15GB model")

    monkeypatch.setattr(
        "semantic_kinematics.embeddings.nv_embed_adapter._ensure_all_tied_weights_keys_default",
        _fake_ensure,
    )
    monkeypatch.setattr(
        transformers.AutoModel, "from_pretrained", _fake_from_pretrained
    )

    adapter = NVEmbedAdapter()
    with pytest.raises(RuntimeError, match="stop before actually loading"):
        adapter._load_model()

    assert calls == ["ensure", "from_pretrained"]
