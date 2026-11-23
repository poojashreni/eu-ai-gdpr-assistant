from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from assistant.api import create_app
from assistant.config import Settings
from assistant.models import Answer, Chunk

REVIEW_503 = "The demo has reached its usage limit, please try again later."
RATE_429 = "Too many questions, please try again later."
CAP_429 = "The demo has reached its daily limit, please try again tomorrow."


def make_answer(text="Because of Art. 5 [1]."):
    return Answer(
        answer=text, language="en", citations=[], refused=False,
        latency_ms=1, input_tokens=2, output_tokens=3,
    )  # fmt: skip


class FakeAnswerFn:
    def __init__(self):
        self.calls = []

    def __call__(self, question):
        self.calls.append(question)
        return make_answer(f"answer {len(self.calls)}")


class Clock:
    def __init__(self, when="2026-11-23 12:00:00"):
        self.now = datetime.fromisoformat(when).replace(tzinfo=UTC).timestamp()

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def chunks_file(tmp_path):
    path = tmp_path / "chunks.jsonl"
    chunk = Chunk(
        id="a", text="t", regulation="GDPR", language="en", type="article", number="5",
        source_url="https://eur-lex.europa.eu/x", celex="c", version_date="2016-05-04",
    )  # fmt: skip
    path.write_text((chunk.model_dump_json() + "\n") * 2, encoding="utf-8")
    return path


def make_client(answer_fn=None, clock=None, **settings):
    settings = Settings(_env_file=None, **settings)
    app = create_app(answer_fn or FakeAnswerFn(), settings, clock=clock or Clock())
    return TestClient(app)


def test_ask_returns_answer():
    client = make_client()
    response = client.post("/ask", json={"question": "  What is Art. 5?  "})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "answer 1"
    assert body["refused"] is False and body["language"] == "en"


def test_rejects_empty_and_too_long():
    fn = FakeAnswerFn()
    client = make_client(fn)
    for question in ("", "   ", "x" * 1001):
        response = client.post("/ask", json={"question": question})
        assert response.status_code == 422, question
        assert isinstance(response.json()["detail"], str)
    assert "1000" in client.post("/ask", json={"question": "x" * 1001}).json()["detail"]
    assert client.post("/ask", json={"question": "x" * 1000}).status_code == 200
    assert len(fn.calls) == 1  # the rejected ones never reached the model


def test_rate_limit_per_ip():
    client = make_client()
    for i in range(10):
        assert client.post("/ask", json={"question": f"q{i}"}).status_code == 200
    response = client.post("/ask", json={"question": "q11"})
    assert response.status_code == 429
    assert response.json() == {"detail": RATE_429}


def test_rate_limit_window_slides():
    clock = Clock()
    client = make_client(clock=clock)
    for i in range(10):
        assert client.post("/ask", json={"question": f"q{i}"}).status_code == 200
        clock.advance(60)  # the first request is 600 s old after the loop
    assert client.post("/ask", json={"question": "late"}).status_code == 429
    clock.advance(3000)  # 3600 s since the first request: it leaves the window
    assert client.post("/ask", json={"question": "later"}).status_code == 200
    assert client.post("/ask", json={"question": "later 2"}).status_code == 429


def test_x_forwarded_for_first_hop():
    client = make_client(rate_limit_per_hour=1)
    headers_a = {"X-Forwarded-For": "1.1.1.1, 10.0.0.1"}
    headers_b = {"X-Forwarded-For": "2.2.2.2, 10.0.0.1"}
    assert client.post("/ask", json={"question": "q1"}, headers=headers_a).status_code == 200
    assert client.post("/ask", json={"question": "q2"}, headers=headers_a).status_code == 429
    assert client.post("/ask", json={"question": "q3"}, headers=headers_b).status_code == 200
    # Same first hop behind a different proxy is still the same client.
    other_proxy = {"X-Forwarded-For": "1.1.1.1, 10.9.9.9"}
    assert client.post("/ask", json={"question": "q4"}, headers=other_proxy).status_code == 429
    # Without the header the socket address is used (a third client).
    assert client.post("/ask", json={"question": "q5"}).status_code == 200


def test_daily_cap():
    client = make_client(daily_cap=2)
    assert client.post("/ask", json={"question": "q1"}).status_code == 200
    assert client.post("/ask", json={"question": "q2"}).status_code == 200
    response = client.post("/ask", json={"question": "q3"})
    assert response.status_code == 429
    assert response.json() == {"detail": CAP_429}
    # A cached answer needs no model call, so it is still served at the cap.
    assert client.post("/ask", json={"question": "q1"}).status_code == 200


def test_daily_cap_resets_at_utc_midnight():
    clock = Clock("2026-11-23 23:59:00")
    client = make_client(clock=clock, daily_cap=1, rate_limit_per_hour=100)
    assert client.post("/ask", json={"question": "q1"}).status_code == 200
    assert client.post("/ask", json={"question": "q2"}).status_code == 429
    clock.advance(120)  # 00:01 UTC the next day
    assert client.post("/ask", json={"question": "q3"}).status_code == 200


def test_cache_hit_does_not_call_answer_fn_twice():
    fn = FakeAnswerFn()
    client = make_client(fn)
    first = client.post("/ask", json={"question": "What is Art. 5?"}).json()
    second = client.post("/ask", json={"question": "  what IS art. 5?  "}).json()
    assert first == second
    assert len(fn.calls) == 1


def test_cached_answers_count_toward_ip_limit():
    client = make_client(rate_limit_per_hour=2)
    assert client.post("/ask", json={"question": "same"}).status_code == 200
    assert client.post("/ask", json={"question": "same"}).status_code == 200
    assert client.post("/ask", json={"question": "same"}).status_code == 429


def test_provider_error_returns_503_and_is_not_cached():
    calls = []

    def flaky(question):
        calls.append(question)
        if len(calls) == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED key=SECRET-KEY-VALUE")
        return make_answer()

    client = make_client(flaky)
    response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json() == {"detail": REVIEW_503}
    assert "SECRET" not in response.text
    assert client.post("/ask", json={"question": "q"}).status_code == 200  # not cached
    assert len(calls) == 2


def test_provider_error_log_has_type_only(caplog):
    def broken(question):
        raise RuntimeError("secret-detail-xyz")

    client = make_client(broken)
    with caplog.at_level("WARNING"):
        client.post("/ask", json={"question": "q"})
    assert "RuntimeError" in caplog.text
    assert "secret-detail-xyz" not in caplog.text


def test_health(chunks_file):
    client = make_client(chunks_path=chunks_file)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["chunks"] == 2
    assert body["text_as_of"] == "AI Act 2026-07-27; GDPR 2016-05-04"


def test_app_without_key_starts_and_ask_returns_503(monkeypatch, chunks_file):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    app = create_app(settings=Settings(_env_file=None, chunks_path=chunks_file))
    with TestClient(app) as client:  # runs the lifespan
        assert client.get("/health").json()["chunks"] == 2
        response = client.post("/ask", json={"question": "q"})
    assert response.status_code == 503
    assert response.json() == {"detail": REVIEW_503}
