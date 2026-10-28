"""Regression guard for issue #34: no hardcoded user-home paths.

Hardcoded ``/home/shane`` paths previously blocked the sentence_transformers
backend and the null-building scripts under any non-``shane`` user. These tests
assert (a) the literal is gone from the touched modules and (b) path resolution
is environment-driven with a working default.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

# Modules touched by the #34 fix and the source files behind them.
from scripts import build_conditioned_null, build_displacement_null, smoke_jolt
from semantic_kinematics.embeddings import sentence_transformers_adapter

_TOUCHED_MODULES = [
    sentence_transformers_adapter,
    build_displacement_null,
    build_conditioned_null,
    smoke_jolt,
]

VAULT_ENV = "THOUGHT_VAULT_VECTORS_DIR"
CANONICAL_DB_ENV = "THOUGHT_VAULT_CANONICAL_DB"
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
    "module",
    [build_displacement_null, build_conditioned_null],
)
def test_null_builder_canonical_db_env_override(monkeypatch, module):
    """THOUGHT_VAULT_CANONICAL_DB drives both live TVI null builders."""
    monkeypatch.setenv(CANONICAL_DB_ENV, "/srv/data/vault/canonical/corpus.db")
    importlib.reload(module)
    try:
        resolved = module.DEFAULT_CORPUS_DB
        assert resolved == "/srv/data/vault/canonical/corpus.db"
        assert "/home/shane" not in resolved
    finally:
        monkeypatch.delenv(CANONICAL_DB_ENV, raising=False)
        importlib.reload(module)


def test_smoke_jolt_vault_path_env_override(monkeypatch):
    """The still-vector-bound smoke tool retains its vectors-dir override."""
    monkeypatch.setenv(VAULT_ENV, "/srv/data/vault/vectors")
    importlib.reload(smoke_jolt)
    try:
        assert smoke_jolt.VAULT_DIR == "/srv/data/vault/vectors"
    finally:
        monkeypatch.delenv(VAULT_ENV, raising=False)
        importlib.reload(smoke_jolt)


def test_embed_wrapper_defaults_to_canonical_db_not_legacy_chunks_jsonl():
    wrapper = Path(__file__).parents[1] / "scripts" / "embed_full_corpus.sh"
    source = wrapper.read_text(encoding="utf-8")

    assert "output/canonical/corpus.db" in source
    assert "output/vectors/chunks.jsonl" not in source
    assert "/home/" not in source
    assert 'CORPUS="${CORPUS:-' not in source  # generic mode must be explicit


def test_vault_path_default_rooted_outside_home():
    """Default TVI paths are rooted at /srv, not a user home directory."""
    for module, attr in [
        (build_displacement_null, "DEFAULT_CORPUS_DB"),
        (build_conditioned_null, "DEFAULT_CORPUS_DB"),
        (smoke_jolt, "VAULT_DIR"),
    ]:
        importlib.reload(module)
        value = getattr(module, attr)
        assert not value.startswith("/home/"), f"{module.__name__}.{attr} = {value}"
        assert value.startswith("/srv/")
