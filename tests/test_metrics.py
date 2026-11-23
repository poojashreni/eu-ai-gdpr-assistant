import math

import pytest

from assistant.models import Answer, Citation
from eval.metrics import citation_correct, hit_at_k, percentile, refusal_correct

ART6_2 = [{"regulation": "AI_ACT", "type": "article", "number": "6", "paragraph": "2"}]


def answer(refused, citations=()):
    return Answer(
        answer="x",
        language="en",
        citations=list(citations),
        refused=refused,
        latency_ms=1,
        input_tokens=0,
        output_tokens=0,
    )


def citation(number, paragraph=None, reg="AI_ACT", typ="article"):
    return Citation(
        regulation=reg,
        type=typ,
        number=number,
        paragraph=paragraph,
        source_url="https://eur-lex.europa.eu/x",
        language="en",
    )


def test_hit_at_k_article_level():
    retrieved = [("GDPR", "article", "6", "1"), ("AI_ACT", "article", "6", "3")]
    assert hit_at_k(retrieved, ART6_2, "article") is True
    assert hit_at_k(retrieved, ART6_2, "paragraph") is False
    assert hit_at_k([("AI_ACT", "article", "6", "2")], ART6_2, "paragraph") is True


def test_hit_at_k_requires_same_regulation_and_type():
    assert hit_at_k([("GDPR", "article", "6", "2")], ART6_2, "article") is False
    assert hit_at_k([("AI_ACT", "recital", "6", None)], ART6_2, "article") is False


def test_hit_at_k_paragraph_level_without_expected_paragraph_is_article_match():
    expected = [{"regulation": "GDPR", "type": "recital", "number": "26", "paragraph": None}]
    assert hit_at_k([("GDPR", "recital", "26", None)], expected, "paragraph") is True


def test_hit_at_k_any_of_several_expected_and_empty_lists():
    art22 = {"regulation": "GDPR", "type": "article", "number": "22", "paragraph": None}
    expected = ART6_2 + [art22]
    assert hit_at_k([("GDPR", "article", "22", "1")], expected, "article") is True
    assert hit_at_k([], expected, "article") is False
    assert hit_at_k([("GDPR", "article", "22", "1")], [], "article") is False


def test_citation_correct():
    assert citation_correct([citation("6", "3")], ART6_2) is True  # article level
    assert citation_correct([citation("5"), citation("6", "2")], ART6_2) is True
    assert citation_correct([citation("6", reg="GDPR")], ART6_2) is False
    assert citation_correct([], ART6_2) is False


def test_refusal_correct():
    assert refusal_correct(answer(refused=True), out_of_scope=True) is True
    assert refusal_correct(answer(refused=False), out_of_scope=False) is True
    assert refusal_correct(answer(refused=False), out_of_scope=True) is False
    assert refusal_correct(answer(refused=True), out_of_scope=False) is False


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10], 95) == 10
    assert percentile([3, 1, 2], 0) == 1 and percentile([3, 1, 2], 100) == 3
    assert math.isclose(percentile(list(range(1, 101)), 95), 95.05)
    with pytest.raises(ValueError):
        percentile([], 50)
