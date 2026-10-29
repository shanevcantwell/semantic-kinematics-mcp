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

import builtins
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest

import scripts.embed_corpus as ec
import scripts.tvi_canonical_input as canonical


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


def test_main_lmstudio_model_env_fallback(tmp_path, monkeypatch):
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
        monkeypatch.setattr(
            ec,
            "load_corpus_source",
            lambda **kwargs: ec.CorpusSource(
                [("hi", "1")],
                {"source_kind": "generic_jsonl", "path": "/tmp/corpus", "sha256": "1" * 64},
            ),
        )
        monkeypatch.setattr(ec, "reconcile_source_sidecar", lambda *a, **k: None)
        monkeypatch.setattr(ec, "source_summary", lambda provenance: "source_kind=generic_jsonl")

        with pytest.raises(SystemExit):
            ec.main(
                [
                    str(tmp_path / "corpus.jsonl"),
                    "--checkpoint",
                    str(tmp_path / "out.jsonl"),
                    "--backend",
                    "lmstudio",
                ]
            )

        assert captured["model"] == "some-model"
        assert captured["base_url"] == "http://host:8082/v1"
    finally:
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        monkeypatch.delenv("EMBEDDING_SERVER_URL", raising=False)
        importlib.reload(ec)


def _canonical_provenance():
    data = {
        "query_contract_version": canonical.QUERY_CONTRACT_VERSION,
        "projection_version": canonical.CHUNK_PROJECTION_VERSION,
        "generation_id": f"tvi-generation-v1:sha256:{'1' * 64}",
        "database_sha256": "2" * 64,
        "logical_digest_sha256": "3" * 64,
        "verification_report_id": f"tvi-verification-v1:sha256:{'4' * 64}",
        "verified_open": True,
        "query": {
            "verb": canonical.CHUNK_QUERY_VERB,
            "ordering": canonical.CHUNK_ORDERING,
            "query_fingerprint_sha256": "5" * 64,
        },
    }
    return SimpleNamespace(as_dict=lambda: data)


class _FakeCanonicalStream:
    def __init__(self, rows):
        self.rows = rows
        self.provenance = _canonical_provenance()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def __iter__(self):
        return iter(self.rows)


def test_canonical_items_match_jsonl_blank_filter_and_order(tmp_path, monkeypatch):
    rows = [
        {"chunk_id": "a", "text": "alpha"},
        {"chunk_id": "blank", "text": "\u2003\t"},
        {"chunk_id": "b", "text": " beta "},
    ]
    monkeypatch.setattr(
        ec,
        "open_canonical_chunks",
        lambda *args, **kwargs: _FakeCanonicalStream(rows),
    )
    canonical_source = ec.load_corpus_source(
        corpus_db=str(tmp_path / "corpus.db"), input_keys=("base",)
    )

    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        '{"chunk_id":"a","text":"alpha"}\n'
        '{"chunk_id":"blank","text":"\\u2003\\t"}\n'
        '{"chunk_id":"b","text":" beta "}\n',
        encoding="utf-8",
    )
    jsonl_source = ec.load_corpus_source(corpus=str(jsonl))

    assert canonical_source.items == jsonl_source.items == [
        ("alpha", "a"),
        (" beta ", "b"),
    ]
    assert canonical_source.provenance["source_kind"] == "tvi_canonical_db"
    assert jsonl_source.provenance["source_kind"] == "generic_jsonl"
    assert canonical_source.provenance["normalized_input_keys"] == ["base"]


def test_input_modes_are_mutually_exclusive(tmp_path):
    with pytest.raises(ValueError, match="exactly one source"):
        ec.load_corpus_source()
    with pytest.raises(ValueError, match="exactly one source"):
        ec.load_corpus_source(
            corpus=str(tmp_path / "corpus.jsonl"),
            corpus_db=str(tmp_path / "corpus.db"),
            input_keys=("base",),
        )
    with pytest.raises(ValueError, match="at least one --input-key"):
        ec.load_corpus_source(corpus_db=str(tmp_path / "corpus.db"))


def test_canonical_failure_does_not_fallback_to_jsonl(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise canonical.CanonicalQueryError("verified-open rejected")

    monkeypatch.setattr(ec, "open_canonical_chunks", fail)
    monkeypatch.setattr(
        ec,
        "_read_items",
        lambda *args, **kwargs: pytest.fail("canonical failure silently fell back"),
    )

    with pytest.raises(canonical.CanonicalQueryError, match="verified-open rejected"):
        ec.load_corpus_source(
            corpus_db=str(tmp_path / "corpus.db"), input_keys=("base",)
        )


def test_source_sidecar_is_atomic_and_refuses_generation_or_query_mismatch(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.jsonl"
    provenance = _canonical_provenance().as_dict() | {
        "source_kind": "tvi_canonical_db",
        "normalized_input_keys": ["base"],
    }
    replacements = []
    real_replace = ec.os.replace

    def recording_replace(source, destination):
        replacements.append((source, destination))
        real_replace(source, destination)

    monkeypatch.setattr(ec.os, "replace", recording_replace)
    ec.reconcile_source_sidecar(str(checkpoint), provenance)

    sidecar = tmp_path / "checkpoint.jsonl.source.json"
    assert sidecar.is_file()
    assert replacements and replacements[0][1] == str(sidecar)
    assert list(tmp_path.glob("checkpoint.jsonl.source.json.*")) == [
        tmp_path / "checkpoint.jsonl.source.json.lock"
    ]

    changed_generation = dict(provenance, generation_id="different-generation")
    with pytest.raises(ValueError, match="source mismatch"):
        ec.reconcile_source_sidecar(str(checkpoint), changed_generation)

    changed_query = dict(provenance)
    changed_query["query"] = dict(provenance["query"], query_fingerprint_sha256="9" * 64)
    with pytest.raises(ValueError, match="source mismatch"):
        ec.reconcile_source_sidecar(str(checkpoint), changed_query)


def _generic_provenance(*, sha256: str = "1" * 64):
    return {
        "source_kind": "generic_jsonl",
        "path": "/tmp/corpus.jsonl",
        "sha256": sha256,
        "text_field": "text",
        "id_field": "chunk_id",
    }


def test_populated_legacy_checkpoint_without_source_sidecar_is_refused(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"chunk_id":"a","embedding":[1]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="populated but has no source provenance"):
        ec.reconcile_source_sidecar(str(checkpoint), _generic_provenance())

    assert not (tmp_path / "checkpoint.jsonl.source.json").exists()


def test_empty_checkpoint_can_receive_first_source_sidecar(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.touch()

    ec.reconcile_source_sidecar(str(checkpoint), _generic_provenance())

    assert (tmp_path / "checkpoint.jsonl.source.json").is_file()


@pytest.mark.parametrize("checkpoint_state", ["absent", "empty"])
def test_legacy_jsonl_current_source_cannot_create_sidecar(tmp_path, checkpoint_state):
    checkpoint = tmp_path / "checkpoint.jsonl"
    if checkpoint_state == "empty":
        checkpoint.touch()
    legacy_current = dict(_generic_provenance(), source_kind="legacy_jsonl")

    with pytest.raises(ValueError, match="stored-sidecar compatibility only"):
        ec.reconcile_source_sidecar(str(checkpoint), legacy_current)

    assert not (tmp_path / "checkpoint.jsonl.source.json").exists()
    assert not (tmp_path / "checkpoint.jsonl.source.json.lock").exists()


def test_source_summary_rejects_legacy_jsonl_current_provenance():
    legacy_current = dict(_generic_provenance(), source_kind="legacy_jsonl")

    with pytest.raises(ValueError, match="stored-sidecar compatibility only"):
        ec.source_summary(legacy_current)


def test_populated_checkpoint_accepts_matching_legacy_jsonl_sidecar(tmp_path):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"alpha"}\n', encoding="utf-8")
    source = ec.load_corpus_source(corpus=str(corpus))
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"chunk_id":"a","embedding":[1]}\n', encoding="utf-8")
    sidecar = tmp_path / "checkpoint.jsonl.source.json"
    legacy_provenance = dict(source.provenance, source_kind="legacy_jsonl")
    sidecar.write_text(json.dumps(legacy_provenance), encoding="utf-8")

    reconciled = ec.reconcile_source_sidecar(str(checkpoint), source.provenance)

    assert reconciled["source_kind"] == "generic_jsonl"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == legacy_provenance
    assert ec.source_identity(reconciled) == ec.source_identity(source.provenance)


def test_legacy_jsonl_current_source_is_rejected_before_sidecar_comparison(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    sidecar = tmp_path / "checkpoint.jsonl.source.json"
    legacy_current = dict(_generic_provenance(), source_kind="legacy_jsonl")
    sidecar.write_text(json.dumps(legacy_current), encoding="utf-8")

    with pytest.raises(ValueError, match="stored-sidecar compatibility only"):
        ec.reconcile_source_sidecar(str(checkpoint), legacy_current)

    assert json.loads(sidecar.read_text(encoding="utf-8")) == legacy_current


@pytest.mark.parametrize(
    ("field", "mismatched_value"),
    [
        ("path", "/tmp/different-corpus.jsonl"),
        ("sha256", "0" * 64),
        ("text_field", "body"),
        ("id_field", "id"),
    ],
)
def test_legacy_jsonl_sidecar_refuses_generic_identity_mismatch(
    tmp_path, field, mismatched_value
):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"alpha"}\n', encoding="utf-8")
    source = ec.load_corpus_source(corpus=str(corpus))
    checkpoint = tmp_path / "checkpoint.jsonl"
    checkpoint.write_text('{"chunk_id":"a","embedding":[1]}\n', encoding="utf-8")
    sidecar = tmp_path / "checkpoint.jsonl.source.json"
    legacy_provenance = dict(source.provenance, source_kind="legacy_jsonl")
    legacy_provenance[field] = mismatched_value
    sidecar.write_text(json.dumps(legacy_provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="source mismatch"):
        ec.reconcile_source_sidecar(str(checkpoint), source.provenance)


def test_conflicting_sidecar_writer_waits_for_run_lock_then_refuses(tmp_path):
    checkpoint = tmp_path / "checkpoint.jsonl"
    first = _generic_provenance(sha256="1" * 64)
    second = _generic_provenance(sha256="2" * 64)
    started = threading.Event()
    finished = threading.Event()
    errors = []

    def reconcile_conflict():
        started.set()
        try:
            ec.reconcile_source_sidecar(str(checkpoint), second)
        except ValueError as exc:  # captured for assertion in the main thread
            errors.append(exc)
        finally:
            finished.set()

    with ec.checkpoint_lock(str(checkpoint), exclusive=True):
        ec.reconcile_source_sidecar(str(checkpoint), first, lock_held=True)
        writer = threading.Thread(target=reconcile_conflict)
        writer.start()
        assert started.wait(timeout=1)
        assert not finished.wait(timeout=0.1)

    writer.join(timeout=1)
    assert not writer.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)
    assert "source mismatch" in str(errors[0])
    sidecar = json.loads(
        (tmp_path / "checkpoint.jsonl.source.json").read_text(encoding="utf-8")
    )
    assert sidecar["sha256"] == "1" * 64


def test_main_holds_exclusive_checkpoint_lock_through_embedding(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"chunk_id":"a","text":"alpha"}\n', encoding="utf-8")
    checkpoint = tmp_path / "checkpoint.jsonl"
    competing_lock_acquired = threading.Event()
    competing_thread = None

    class FakeBulkEmbedder:
        def __init__(self, adapter, **kwargs):
            assert kwargs["checkpoint_path"] == str(checkpoint)

        def embed_corpus(self, items):
            nonlocal competing_thread

            def acquire_competing_lock():
                with ec.checkpoint_lock(str(checkpoint), exclusive=True):
                    competing_lock_acquired.set()

            competing_thread = threading.Thread(target=acquire_competing_lock)
            competing_thread.start()
            assert not competing_lock_acquired.wait(timeout=0.1)
            return {}

    monkeypatch.setattr(ec, "_make_adapter", lambda *args: object())
    monkeypatch.setattr(ec, "BulkEmbedder", FakeBulkEmbedder)

    ec.main(
        [
            str(corpus),
            "--checkpoint",
            str(checkpoint),
            "--backend",
            "sentence_transformers",
        ]
    )

    assert competing_lock_acquired.wait(timeout=1)
    assert competing_thread is not None
    competing_thread.join(timeout=1)
    assert not competing_thread.is_alive()


def test_generic_jsonl_items_and_hash_come_from_one_open_stream(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    raw = b'{"chunk_id":"a","text":"alpha"}\n'
    corpus.write_bytes(raw)
    real_open = builtins.open
    source_opens = 0

    def counting_open(path, mode="r", *args, **kwargs):
        nonlocal source_opens
        if str(path) == str(corpus) and mode == "rb":
            source_opens += 1
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    source = ec.load_corpus_source(corpus=str(corpus))

    assert source.items == [("alpha", "a")]
    assert source.provenance["source_kind"] == "generic_jsonl"
    assert source.provenance["sha256"] == hashlib.sha256(raw).hexdigest()
    assert source_opens == 1


def test_generic_jsonl_mutation_during_read_is_refused(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        '{"chunk_id":"a","text":"alpha"}\n'
        '{"chunk_id":"b","text":"beta"}\n',
        encoding="utf-8",
    )
    real_open = builtins.open

    class MutatingReader:
        def __init__(self, file_object):
            self.file_object = file_object

        def __enter__(self):
            self.file_object.__enter__()
            return self

        def __exit__(self, *args):
            return self.file_object.__exit__(*args)

        def fileno(self):
            return self.file_object.fileno()

        def __iter__(self):
            for index, line in enumerate(self.file_object):
                yield line
                if index == 0:
                    with real_open(corpus, "ab") as output:
                        output.write(b"\n")

    def mutating_open(path, mode="r", *args, **kwargs):
        opened = real_open(path, mode, *args, **kwargs)
        if str(path) == str(corpus) and mode == "rb":
            return MutatingReader(opened)
        return opened

    monkeypatch.setattr(builtins, "open", mutating_open)

    with pytest.raises(ValueError, match="changed while being read"):
        ec.load_corpus_source(corpus=str(corpus))
