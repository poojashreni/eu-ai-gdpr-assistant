import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from assistant.chain import (
    NOT_COVERED_MESSAGE,
    NOT_COVERED_TOKEN,
    answer_question,
    detect_language,
    format_passages,
    make_llm,
    parse_citations,
)
from assistant.config import Settings


def make_doc(number, paragraph="", typ="article", reg="AI_ACT", lang="en", text="Some text."):
    return Document(
        page_content=f"Article {number}\n{text}",
        metadata={
            "regulation": reg, "type": typ, "number": number, "paragraph": paragraph,
            "title": "", "source_url": f"https://eur-lex.europa.eu/{number}", "language": lang,
        },
    )  # fmt: skip


DOCS = [make_doc("6", "2"), make_doc("5"), make_doc("26", typ="recital", reg="GDPR")]


def stub_retrieve(question, language):
    return DOCS


class RecordingFake(FakeListChatModel):
    """Fake model that remembers the messages it was sent."""

    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        msg = AIMessage(
            content=self.responses[0],
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_parse_citations_maps_markers():
    text, cites = parse_citations("Banned [1]. Also see [3] and again [1].", DOCS)
    assert text == "Banned [1]. Also see [3] and again [1]."
    assert [c.label for c in cites] == ["AI Act Art. 6(2)", "GDPR Recital 26"]
    assert cites[0].paragraph == "2" and cites[0].title is None
    assert cites[0].source_url == "https://eur-lex.europa.eu/6"


def test_invalid_markers_dropped():
    text, cites = parse_citations("x [9] y [2]", DOCS)
    assert "[9]" not in text
    assert text == "x y [2]"
    assert [c.label for c in cites] == ["AI Act Art. 5"]


def test_grouped_markers_are_normalised():
    text, cites = parse_citations("a [1][3] b [1, 3] c [2, 9]", DOCS)
    assert text == "a [1][3] b [1][3] c [2]"
    assert [c.label for c in cites] == ["AI Act Art. 6(2)", "GDPR Recital 26", "AI Act Art. 5"]


def test_format_passages():
    out = format_passages(DOCS)
    assert out.startswith("[1] AI Act Art. 6(2) (en): Article 6")
    assert "[3] GDPR Recital 26 (en):" in out


def test_detect_language():
    assert detect_language("Welche KI-Systeme sind nach der Verordnung verboten?") == "de"
    assert detect_language("Which AI systems are prohibited under the regulation?") == "en"
    assert detect_language("Quels systèmes d'IA sont interdits par le règlement ?") == "en"
    assert detect_language("Artikel 5?") == "de"
    assert detect_language("Art 5") == "en"
    assert detect_language("") == "en"


@pytest.mark.parametrize("raw", ["NOT_COVERED", "  NOT_COVERED.\n", '"NOT_COVERED"', "not_covered"])
def test_not_covered_token_becomes_refusal(raw):
    llm = FakeListChatModel(responses=[raw])
    ans = answer_question("Wie ist das Wetter in Berlin heute?", stub_retrieve, llm)
    assert ans.refused is True
    assert ans.language == "de"
    assert ans.answer == NOT_COVERED_MESSAGE["de"]
    assert ans.citations == []


def test_uncited_answer_becomes_refusal():
    llm = FakeListChatModel(responses=["Confident answer with no citation markers."])
    ans = answer_question("Which AI practices are banned?", stub_retrieve, llm)
    assert ans.refused is True
    assert ans.answer == NOT_COVERED_MESSAGE["en"]
    assert ans.citations == []


def test_only_invalid_markers_becomes_refusal():
    llm = FakeListChatModel(responses=["Claim [9]."])
    assert answer_question("Which AI practices are banned?", stub_retrieve, llm).refused


def test_cited_answer_passes_through_with_tokens_and_latency():
    llm = RecordingFake(responses=["Some practices are banned [1]."])
    ans = answer_question("Which AI practices are banned?", stub_retrieve, llm)
    assert ans.refused is False
    assert ans.answer == "Some practices are banned [1]."
    assert [c.label for c in ans.citations] == ["AI Act Art. 6(2)"]
    assert (ans.input_tokens, ans.output_tokens) == (7, 3)
    assert isinstance(ans.latency_ms, int) and ans.latency_ms >= 0


def test_tokens_default_to_zero_without_usage_metadata():
    llm = FakeListChatModel(responses=["Yes [2]."])
    ans = answer_question("Is Art. 5 relevant?", stub_retrieve, llm)
    assert (ans.input_tokens, ans.output_tokens) == (0, 0)


def test_non_de_en_question_answers_in_english():
    llm = RecordingFake(responses=["Certaines pratiques sont interdites [1]."])
    llm.seen = []
    question = "Quelles pratiques d'IA sont interdites par le règlement ?"
    ans = answer_question(question, stub_retrieve, llm)
    assert ans.language == "en"
    assert ans.refused is False and len(ans.citations) == 1
    prompt = " ".join(m.content for m in llm.seen[0])
    assert "English" in prompt and "German" not in prompt
    assert NOT_COVERED_TOKEN in prompt
    assert "[1] AI Act Art. 6(2) (en)" in prompt


def test_german_question_asks_for_german():
    llm = RecordingFake(responses=["Antwort [1]."])
    llm.seen = []
    answer_question("Welche KI-Praktiken sind nach der Verordnung verboten?", stub_retrieve, llm)
    prompt = " ".join(m.content for m in llm.seen[0])
    assert "German" in prompt


def test_no_passages_still_refuses_via_model_or_marker_rule():
    llm = FakeListChatModel(responses=["Anything [1]."])
    ans = answer_question("Which AI practices are banned?", lambda q, lang: [], llm)
    assert ans.refused is True


def test_make_llm_requires_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        make_llm(Settings(_env_file=None, llm_provider="gemini"))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        make_llm(Settings(_env_file=None, llm_provider="openai"))
