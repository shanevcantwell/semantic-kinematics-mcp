"""Regression guard for issue #34: no hardcoded user-home paths.

Hardcoded ``/home/shane`` paths previously blocked the sentence_transformers
backend and the null-building scripts under any non-``shane`` user. These tests
assert (a) the literal is gone from the touched modules and (b) path resolution
is environment-driven with a working default.
"""

from __future__ import annotations

import importlib
import os
import re

import pytest

# Modules touched by the #34 fix and the source files behind them.
import scripts.build_conditioned_null as build_conditioned_null
import scripts.build_displacement_null as build_displacement_null
import scripts.smoke_jolt as smoke_jolt
from semantic_kinematics.embeddings import sentence_transformers_adapter

_TOUCHED_MODULES = [
    sentence_transformers_adapter,
    build_displacement_null,
    build_conditioned_null,
    smoke_jolt,
]

VAULT_ENV = "THOUGHT_VAULT_VECTORS_DIR"
MODEL_ENV = "NV_EMBED_MODEL_PATH"


def test_no_home_shane_literal_in_touched_modules():
    """No ``/home/shane`` (or other ``/home/<user>``) literal in source."""
    offenders = []
    for mod in _TOUCHED_MODULES:
        with open(mod.__file__, encoding="utf-8") as fh:
            src = fh.read()
        if re.search(r"/home/\w+", src):
            offenders.append(mod.__file__)
    assert not offenders, f"/home/<user> literal still present in: {offenders}"


def test_adapter_model_path_default_is_user_agnostic(monkeypatch):
    """Default resolves to the HuggingFace hub id, not a local home path."""
    monkeypatch.delenv(MODEL_ENV, raising=False)
    importlib.reload(sentence_transformers_adapter)
    default = sentence_transformers_adapter.SentenceTransformersAdapter.DEFAULT_MODEL_PATH
    assert not default.startswith("/home/")
    assert default == "nvidia/NV-Embed-v2"


def test_adapter_model_path_env_override(monkeypatch):
    """NV_EMBED_MODEL_PATH overrides the default model reference."""
    monkeypatch.setenv(MODEL_ENV, "/srv/models/NV-Embed-v2")
    importlib.reload(sentence_transformers_adapter)
    try:
        default = sentence_transformers_adapter.SentenceTransformersAdapter.DEFAULT_MODEL_PATH
        assert default == "/srv/models/NV-Embed-v2"
    finally:
        # Restore module to the unpatched env so later tests see the real default.
        monkeypatch.delenv(MODEL_ENV, raising=False)
        importlib.reload(sentence_transformers_adapter)


@pytest.mark.parametrize(
    "module, attr, suffix",
    [
        (build_displacement_null, "DEFAULT_SOURCE", "chunks.jsonl"),
        (build_conditioned_null, "DEFAULT_SOURCE", "chunks.jsonl"),
        (smoke_jolt, "VAULT_DIR", ""),
    ],
)
def test_vault_path_env_override(monkeypatch, module, attr, suffix):
    """THOUGHT_VAULT_VECTORS_DIR drives every vault-corpus consumer."""
    monkeypatch.setenv(VAULT_ENV, "/srv/data/vault/vectors")
    importlib.reload(module)
    try:
        resolved = getattr(module, attr)
        expected = os.path.join("/srv/data/vault/vectors", suffix) if suffix else "/srv/data/vault/vectors"
        assert resolved == expected
        assert "/home/shane" not in resolved
    finally:
        monkeypatch.delenv(VAULT_ENV, raising=False)
        importlib.reload(module)


def test_vault_path_default_rooted_outside_home():
    """Default vault dir is rooted at /srv, not a user home directory."""
    for module, attr in [
        (build_displacement_null, "DEFAULT_SOURCE"),
        (build_conditioned_null, "DEFAULT_SOURCE"),
        (smoke_jolt, "VAULT_DIR"),
    ]:
        importlib.reload(module)
        value = getattr(module, attr)
        assert not value.startswith("/home/"), f"{module.__name__}.{attr} = {value}"
        assert value.startswith("/srv/")
