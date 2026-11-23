"""The answer chain: retrieve passages, ask the LLM to answer from them only, parse citations."""

import re
import time
from collections.abc import Callable
from typing import Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langdetect import DetectorFactory, LangDetectException, detect

from assistant.config import Settings
from assistant.models import Answer, Citation

DetectorFactory.seed = 0  # langdetect is randomised by default; a fixed seed makes it repeatable

NOT_COVERED_TOKEN = "NOT_COVERED"
NOT_COVERED_MESSAGE = {
    "en": "This question is not covered by the text of the EU AI Act or the GDPR.",
    "de": "Diese Frage wird vom Text der KI-Verordnung bzw. der DSGVO nicht abgedeckt.",
}

SYSTEM_PROMPT = """\
You are an assistant that answers questions about the EU AI Act and the GDPR.

Rules:
1. Answer ONLY from the numbered passages below. Do not use any other knowledge.
2. Write the answer in {language}.
3. Cite every claim with the number of the passage that supports it, like [1] or [2][3].
   Use only the numbers of the passages given. Never invent a number.
4. If the passages do not contain the answer, reply with exactly {token} and nothing else.
5. Be concise and precise. Do not give legal advice."""

HUMAN_PROMPT = """\
Passages:
{passages}

Question: {question}"""

PROMPT = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])

_GERMAN_CHARS = set("äöüß")
_GERMAN_WORDS = {
    "artikel", "was", "wer", "wie", "wann", "welche", "welcher", "welches", "ist", "sind",
    "darf", "dürfen", "muss", "müssen", "gibt", "und", "oder", "der", "die", "das", "dsgvo",
}  # fmt: skip
_WORD = re.compile(r"\w+")


def detect_language(text: str) -> Literal["de", "en"]:
    """German if langdetect says German, otherwise English (also for French, Spanish, ...)."""
    words = _WORD.findall(text.lower())
    if len(words) < 3:
        # Too short for langdetect to be reliable: look for obvious German hints instead.
        has_umlaut = any(ch in _GERMAN_CHARS for ch in text.lower())
        if has_umlaut or any(w in _GERMAN_WORDS for w in words):
            return "de"
        return "en"
    try:
        return "de" if detect(text) == "de" else "en"
    except LangDetectException:
        return "en"


def make_llm(settings: Settings) -> BaseChatModel:
    """Build the chat model for the configured provider (imports lazily; temperature 0)."""
    if settings.llm_provider == "gemini":
        if not settings.google_api_key:
            raise ValueError("GOOGLE_API_KEY is not set. Add it to your .env file.")
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model, temperature=0, google_api_key=settings.google_api_key
        )
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.openai_model, temperature=0, api_key=settings.openai_api_key)


def _citation(doc: Document) -> Citation:
    m = doc.metadata
    return Citation(
        regulation=m["regulation"],
        type=m["type"],
        number=m["number"],
        paragraph=m.get("paragraph") or None,
        title=m.get("title") or None,
        source_url=m["source_url"],
        language=m["language"],
    )


def format_passages(docs: list[Document]) -> str:
    """Number the passages: `[1] AI Act Art. 6(2) (en): <text>`."""
    lines = []
    for i, doc in enumerate(docs, start=1):
        c = _citation(doc)
        lines.append(f"[{i}] {c.label} ({c.language}): {doc.page_content}")
    return "\n\n".join(lines)


_MARKER = re.compile(r"(\s?)\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(text: str, docs: list[Document]) -> tuple[str, list[Citation]]:
    """Keep valid `[n]` markers (1 <= n <= len(docs)), drop the rest, collect citations.

    `[1, 3]` is rewritten as `[1][3]`. Citations are unique and in order of first use.
    """
    used: list[int] = []

    def fix(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(2))]
        valid = [n for n in numbers if 1 <= n <= len(docs)]
        for n in valid:
            if n not in used:
                used.append(n)
        if not valid:
            return ""  # the whole marker (and the space before it) disappears
        return match.group(1) + "".join(f"[{n}]" for n in valid)

    cleaned = _MARKER.sub(fix, text).strip()
    return cleaned, [_citation(docs[n - 1]) for n in used]


def _is_not_covered(text: str) -> bool:
    return text.strip(" \t\r\n.\"'`*").upper() == NOT_COVERED_TOKEN


def _message_text(message: AIMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    # Some providers return a list of content blocks.
    return "".join(part if isinstance(part, str) else part.get("text", "") for part in content)


def answer_question(
    question: str,
    retrieve: Callable[[str, str | None], list[Document]],
    llm: BaseChatModel,
) -> Answer:
    start = time.perf_counter()
    language = detect_language(question)
    docs = retrieve(question, language)
    chain = PROMPT | llm  # LCEL: the prompt's output feeds straight into the model
    message = chain.invoke(
        {
            "language": "German" if language == "de" else "English",
            "token": NOT_COVERED_TOKEN,
            "passages": format_passages(docs),
            "question": question,
        }
    )
    raw = _message_text(message)
    latency_ms = int((time.perf_counter() - start) * 1000)
    usage = getattr(message, "usage_metadata", None) or {}

    text, citations = ("", [])
    if not _is_not_covered(raw):
        text, citations = parse_citations(raw, docs)
    refused = not citations
    return Answer(
        answer=NOT_COVERED_MESSAGE[language] if refused else text,
        language=language,
        citations=[] if refused else citations,
        refused=refused,
        latency_ms=latency_ms,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
    )
