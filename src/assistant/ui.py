"""Pure logic for the Streamlit page: call the API, turn results and errors into display data."""

from dataclasses import dataclass

import httpx

from assistant.config import get_settings
from assistant.models import Answer

UNAVAILABLE_MESSAGE = "The service is starting up or unavailable, please try again in a minute."
GENERIC_ERROR = "Something went wrong, please try again."
TIMEOUT_SECONDS = 60.0

# (language, question), taken from the in-scope items of eval/gold.jsonl.
EXAMPLES: list[tuple[str, str]] = [
    (
        "DE",
        "Darf ein Arbeitgeber KI einsetzen, um die Emotionen seiner Mitarbeitenden zu erkennen?",
    ),
    ("DE", "Wie schnell muss eine Datenpanne der Aufsichtsbehörde gemeldet werden?"),
    ("EN", "How do I find out whether my AI system counts as high-risk?"),
    ("EN", "Can I make a company delete the personal data it holds about me?"),
]


@dataclass
class AskResult:
    answer: Answer | None = None
    error: str | None = None


def _client() -> httpx.Client:
    return httpx.Client(base_url=get_settings().api_url, timeout=TIMEOUT_SECONDS)


def _error_message(response: httpx.Response) -> str:
    """Map a non-200 API response to a message that is safe to show."""
    if response.status_code not in (422, 429, 503):
        return GENERIC_ERROR
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return GENERIC_ERROR
    if isinstance(detail, str) and detail:
        return detail
    if isinstance(detail, list) and detail and isinstance(detail[0], dict) and detail[0].get("msg"):
        return str(detail[0]["msg"])
    return GENERIC_ERROR


def ask(question: str, client: httpx.Client | None = None) -> AskResult:
    """Send the question to POST /ask; never raises, errors come back as a message."""
    try:
        http = client or _client()
        response = http.post("/ask", json={"question": question})
    except httpx.HTTPError:
        return AskResult(error=UNAVAILABLE_MESSAGE)
    if response.status_code != 200:
        return AskResult(error=_error_message(response))
    try:
        return AskResult(answer=Answer.model_validate(response.json()))
    except ValueError:
        return AskResult(error=GENERIC_ERROR)


def get_text_as_of(client: httpx.Client | None = None) -> str:
    """The consolidation dates of the legal texts, from GET /health, or "unknown"."""
    try:
        http = client or _client()
        response = http.get("/health", timeout=5.0)
        response.raise_for_status()
        return str(response.json()["text_as_of"])
    except (httpx.HTTPError, ValueError, KeyError):
        return "unknown"


def citation_links(answer: Answer) -> list[str]:
    """One markdown line per citation: a link to EUR-Lex, then the provision title."""
    lines = []
    for citation in answer.citations:
        line = f"[{citation.label}]({citation.source_url})"
        if citation.title:
            line += f" - {citation.title} ({citation.language.upper()})"
        lines.append(line)
    return lines


def caption(answer: Answer) -> str:
    return (
        f"{answer.latency_ms / 1000:.1f} s · "
        f"{answer.input_tokens} input / {answer.output_tokens} output tokens"
    )


def footer(text_as_of: str) -> str:
    return (
        "Not legal advice. · Do not enter personal data. · "
        f"This demo is quota-limited. · Text as of: {text_as_of}"
    )
