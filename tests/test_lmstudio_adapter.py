"""Unit tests for LMStudioAdapter. HTTP is mocked; no live server contacted."""

import pytest

from semantic_kinematics.embeddings.lmstudio import LMStudioAdapter


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
    adapter = LMStudioAdapter(base_url="http://localhost:8082/v1")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_tokenize_url_strips_trailing_slash():
    adapter = LMStudioAdapter(base_url="http://localhost:8082/v1/")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_tokenize_url_without_v1():
    adapter = LMStudioAdapter(base_url="http://localhost:8082")
    assert adapter._tokenize_url() == "http://localhost:8082/tokenize"


def test_count_tokens_returns_tokenize_count(monkeypatch):
    """count_tokens returns len of the server's /tokenize token list,
    POSTed to the server-root /tokenize endpoint (not /v1)."""
    adapter = LMStudioAdapter(base_url="http://localhost:8082/v1")

    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        # Server tokenized into 5 tokens.
        return _FakeResponse({"tokens": [1, 2, 3, 4, 5]})

    monkeypatch.setattr(
        "semantic_kinematics.embeddings.lmstudio.requests.post", fake_post
    )

    n = adapter.count_tokens("some dense code: def f(): return {1:2}")
    assert n == 5
    assert captured["url"] == "http://localhost:8082/tokenize"
    assert captured["json"] == {"content": "some dense code: def f(): return {1:2}"}


def test_count_tokens_raises_on_http_error(monkeypatch):
    adapter = LMStudioAdapter(base_url="http://localhost:8082/v1")

    def fake_post(url, json=None, **kwargs):
        return _FakeResponse({}, status_ok=False)

    monkeypatch.setattr(
        "semantic_kinematics.embeddings.lmstudio.requests.post", fake_post
    )

    with pytest.raises(RuntimeError):
        adapter.count_tokens("x")


def test_count_tokens_raises_valueerror_with_body_on_missing_tokens(monkeypatch):
    """A 200 response lacking 'tokens' must raise a diagnosable ValueError that
    includes the response body, not an opaque KeyError."""
    adapter = LMStudioAdapter(base_url="http://localhost:8082/v1")

    def fake_post(url, json=None, **kwargs):
        return _FakeResponse({"error": "model not loaded"})

    monkeypatch.setattr(
        "semantic_kinematics.embeddings.lmstudio.requests.post", fake_post
    )

    with pytest.raises(ValueError) as excinfo:
        adapter.count_tokens("x")
    msg = str(excinfo.value)
    assert "tokens" in msg
    # The actual body is surfaced for diagnosis.
    assert "model not loaded" in msg


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
