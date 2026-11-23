"""Metrics for the evaluation harness. Pure functions, no models, no I/O.

A "key" is the language-independent identity of a provision: (regulation, type, number,
paragraph), the same tuple as `Chunk.key`. Gold expectations are dicts with those four fields.
"""

from collections.abc import Iterable
from typing import Literal

from assistant.models import Answer, Citation

Key = tuple[str, str, str, str | None]


def _article(regulation: str, typ: str, number: str) -> tuple[str, str, str]:
    return (regulation, typ, number)


def hit_at_k(
    retrieved_keys: Iterable[Key], expected: list[dict], level: Literal["article", "paragraph"]
) -> bool:
    """True if any retrieved provision matches any expected one.

    "article": regulation, type and number must match; the paragraph is ignored, so a
    retrieved Art. 6(3) counts for an expected Art. 6(2).
    "paragraph": the paragraph must match too. An expectation without a paragraph (a recital,
    or an article where the gold set names no paragraph) falls back to the article match.
    """
    keys = list(retrieved_keys)
    for exp in expected:
        art = _article(exp["regulation"], exp["type"], exp["number"])
        for key in keys:
            if _article(*key[:3]) != art:
                continue
            if level == "article" or not exp.get("paragraph") or key[3] == exp["paragraph"]:
                return True
    return False


def citation_correct(citations: list[Citation], expected: list[dict]) -> bool:
    """True if at least one cited provision is an expected one (article level)."""
    keys = [(c.regulation, c.type, c.number, c.paragraph) for c in citations]
    return hit_at_k(keys, expected, "article")


def refusal_correct(answer: Answer, out_of_scope: bool) -> bool:
    """Refusing is right for out-of-scope questions and wrong for in-scope ones."""
    return answer.refused == out_of_scope


def percentile(values: Iterable[float], p: float) -> float:
    """The p-th percentile (0-100) with linear interpolation, like numpy's default."""
    data = sorted(values)
    if not data:
        raise ValueError("percentile() of an empty list")
    pos = (len(data) - 1) * p / 100
    low = int(pos)
    high = min(low + 1, len(data) - 1)
    return data[low] + (data[high] - data[low]) * (pos - low)
