"""Unit tests for LMStudioAdapter. HTTP is mocked; no live server contacted."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from semantic_kinematics.embeddings.bulk import BulkEmbedder
from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter


def _embedding_adapter(responses):
    adapter = LMStudioAdapter(model_name="alias", base_url="http://localhost:8082/v1")
    replies = iter(responses)
    adapter._client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: next(replies))
    )
    return adapter


def _reply(indices, model="actual-model"):
    return SimpleNamespace(
        model=model,
        data=[
            SimpleNamespace(index=i, embedding=[float(n + 1), 1.0]) for n, i in enumerate(indices)
        ],
    )


def test_embed_batch_orders_results_by_response_index():
    adapter = _embedding_adapter([_reply([1, 0])])
    np.testing.assert_array_equal(adapter.embed_batch(["first", "second"]), [[2, 1], [1, 1]])
    assert "actual-model" in adapter.model_name


@pytest.mark.parametrize("indices", [[0, 0], [0], [0, 2], [-1, 0], [0, 1.0], [0, True], [0, None]])
def test_embed_batch_rejects_invalid_indices(indices):
    adapter = _embedding_adapter([_reply(indices)])
    with pytest.raises(ValueError, match="index"):
        adapter.embed_batch(["first", "second"])


def test_rejects_model_change_between_calls():
    adapter = _embedding_adapter([_reply([0], "model-a"), _reply([0], "model-b")])
    adapter.embed("first")
    with pytest.raises(ValueError, match="model"):
        adapter.embed_batch(["second"])


@pytest.mark.parametrize("models", [(None, "model-a"), ("model-a", None)])
def test_rejects_response_model_appearing_or_disappearing(models):
    adapter = _embedding_adapter([_reply([0], models[0]), _reply([0], models[1])])
    adapter.embed("first")
    with pytest.raises(ValueError, match="model"):
        adapter.embed("second")


def test_single_embedding_rejects_invalid_index():
    adapter = _embedding_adapter([_reply([1])])
    with pytest.raises(ValueError, match="index"):
        adapter.embed("first")


def test_http_response_order_is_preserved_by_index():
    import httpx
    from openai import OpenAI

    def respond(request):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "actual-model",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [1.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [2.0, 1.0]},
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    adapter = LMStudioAdapter(model_name="alias", base_url="http://test/v1")
    adapter._client = OpenAI(
        base_url="http://test/v1",
        api_key="unused",
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    )
    np.testing.assert_array_equal(adapter.embed_batch(["first", "second"]), [[2, 1], [1, 1]])


def test_checkpoint_records_served_model(tmp_path):
    adapter = _embedding_adapter([_reply([0], "model-a")])
    checkpoint = str(tmp_path / "checkpoint.jsonl")
    bulk = BulkEmbedder(adapter, checkpoint_path=checkpoint)
    bulk.embed_corpus([])
    with open(checkpoint + ".meta.json") as metadata:
        assert "model-a" in json.load(metadata)["model_name"]


class _FakeResponse:
    def __init__(self, payload, status_ok=True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


def test_tokenize_url_strips_v1():
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082/v1")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_tokenize_url_strips_trailing_slash():
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082/v1/")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_tokenize_url_without_v1():
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_count_tokens_returns_tokenize_count(monkeypatch):
    """count_tokens returns len of the server's /tokenize token list,
    POSTed to the server-root /tokenize endpoint (not /v1)."""
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082/v1")

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        # Server tokenized into 5 tokens.
        return _FakeResponse({"tokens": [1, 2, 3, 4, 5]})

    monkeypatch.setattr("semantic_kinematics.embeddings.lmstudio.requests.post", fake_post)

    n = adapter.count_tokens("some dense code: def f(): return {1:2}")
    assert n == 5
    assert captured["url"] == "http://localhost:8082/tokenize"
    assert captured["json"] == {"content": "some dense code: def f(): return {1:2}"}


def test_count_tokens_raises_on_http_error(monkeypatch):
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082/v1")

    def fake_post(url, json=None, **kwargs):
        return _FakeResponse({}, status_ok=False)

    monkeypatch.setattr("semantic_kinematics.embeddings.lmstudio.requests.post", fake_post)

    with pytest.raises(RuntimeError):
        adapter.count_tokens("x")


def test_count_tokens_raises_valueerror_with_body_on_missing_tokens(monkeypatch):
    """A 200 response lacking 'tokens' must raise a diagnosable ValueError that
    includes the response body, not an opaque KeyError."""
    adapter = LMStudioAdapter(model_name="test-model", base_url="http://localhost:8082/v1")

    def fake_post(url, json=None, **kwargs):
        return _FakeResponse({"error": "model not loaded"})

    monkeypatch.setattr("semantic_kinematics.embeddings.lmstudio.requests.post", fake_post)

    with pytest.raises(ValueError) as excinfo:
        adapter.count_tokens("x")
    msg = str(excinfo.value)
    assert "tokens" in msg
    # The actual body is surfaced for diagnosis.
    assert "model not loaded" in msg


def test_constructing_without_model_name_raises():
    """Rule #14: no baked nomic default. Omitting model_name must raise loudly
    rather than implicitly selecting a model."""
    with pytest.raises(ValueError) as excinfo:
        LMStudioAdapter(base_url="http://localhost:8082/v1")
    assert "model" in str(excinfo.value).lower()


def test_constructing_without_base_url_raises():
    """Omitting base_url must raise rather than defaulting to the LM-Studio
    localhost:1234 endpoint."""
    with pytest.raises(ValueError) as excinfo:
        LMStudioAdapter(model_name="test-model")
    assert "endpoint" in str(excinfo.value).lower() or "base_url" in str(excinfo.value)


def test_constructing_with_no_args_raises():
    with pytest.raises(ValueError):
        LMStudioAdapter()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
