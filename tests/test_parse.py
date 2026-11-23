from pathlib import Path

import pytest

from assistant.models import Chunk
from assistant.parse import parse_all, parse_regulation, read_chunks, write_chunks

FIXTURES = Path(__file__).parent / "fixtures"
URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:02024R1689-20260727"
META = dict(
    regulation="AI_ACT",
    language="en",
    source_url=URL,
    celex="02024R1689-20260727",
    version_date="2026-07-27",
)


@pytest.fixture(scope="module")
def consolidated() -> list[Chunk]:
    html = (FIXTURES / "mini_regulation_en.html").read_text(encoding="utf-8")
    return parse_regulation(html, **META)


@pytest.fixture(scope="module")
def recitals() -> list[Chunk]:
    html = (FIXTURES / "mini_original_en.html").read_text(encoding="utf-8")
    return parse_regulation(html, role="recitals", **META)


def by_id(chunks: list[Chunk]) -> dict[str, Chunk]:
    return {c.id: c for c in chunks}


def test_parses_articles_into_paragraph_chunks(consolidated):
    art2 = [c for c in consolidated if c.type == "article" and c.number == "2"]
    assert [c.paragraph for c in art2] == ["1", "2"]
    assert all(c.title == "Scope" for c in art2)
    assert art2[0].id == "AI_ACT-en-article-2-1"
    # The sub-list stays with its paragraph; the unnumbered trailing sentence joins paragraph 2.
    assert "(a) providers placing AI systems on the market;" in art2[0].text
    assert "(b) deployers of AI systems." in art2[0].text
    assert art2[1].text.endswith("This sentence belongs to paragraph 2.")


def test_short_article_single_chunk(consolidated):
    art1 = [c for c in consolidated if c.type == "article" and c.number == "1"]
    assert len(art1) == 1
    assert art1[0].paragraph is None
    assert art1[0].id == "AI_ACT-en-article-1"
    assert art1[0].title == "Subject matter"  # stray trailing apostrophe in the source is dropped


def test_suffixed_article_number_is_kept_as_string(consolidated):
    assert "2a" in {c.number for c in consolidated if c.type == "article"}


def test_recitals_and_annex_points(consolidated, recitals):
    assert [(c.type, c.number, c.paragraph) for c in recitals] == [
        ("recital", "1", None),
        ("recital", "2", None),
    ]
    annex = [c for c in consolidated if c.type == "annex"]
    assert [(c.number, c.paragraph) for c in annex] == [("I", None), ("I", "1"), ("I", "2")]
    assert annex[1].text == (
        "Annex I – High-risk areas, point 1\nBiometrics:\n(a) remote identification systems."
    )
    assert annex[0].text == "Annex I – High-risk areas\nThe following areas are covered:"
    assert annex[1].id == "AI_ACT-en-annex-I-1"
    assert recitals[0].text.startswith("Recital 1\n")


def test_role_selects_what_is_extracted(consolidated, recitals):
    assert {c.type for c in consolidated} == {"article", "annex"}
    assert {c.type for c in recitals} == {"recital"}  # original-text article is ignored


def test_chunk_text_includes_heading(consolidated):
    art2 = [c for c in consolidated if c.type == "article" and c.number == "2"]
    assert art2[0].text.startswith("Article 2 – Scope\n(1) This Regulation applies to:")
    assert consolidated[0].text.startswith("Article 1 – Subject matter\n")


def test_text_is_clean(consolidated, recitals):
    for c in consolidated + recitals:
        assert "\xa0" not in c.text
        assert "▼" not in c.text
        assert "  " not in c.text
    # Footnote markers like "(1)" are removed, other text kept.
    assert (
        recitals[0].text
        == "Recital 1\nThe purpose of this Regulation is to improve the internal market."
    )
    assert "military purposes" in by_id(consolidated)["AI_ACT-en-article-2-2"].text


def test_source_url_gets_anchor(consolidated, recitals):
    ids = by_id(consolidated)
    assert ids["AI_ACT-en-article-2-1"].source_url == f"{URL}#art_2"
    assert ids["AI_ACT-en-article-2a"].source_url == f"{URL}#art_2a"
    assert ids["AI_ACT-en-annex-I-1"].source_url == f"{URL}#anx_I"
    assert recitals[1].source_url == f"{URL}#rct_2"
    assert all(
        c.celex == "02024R1689-20260727" and c.version_date == "2026-07-27" for c in consolidated
    )


def test_ids_unique(consolidated, recitals):
    ids = [c.id for c in consolidated + recitals]
    assert len(ids) == len(set(ids))


def test_write_and_read_roundtrip(consolidated, tmp_path):
    path = tmp_path / "out" / "chunks.jsonl"
    write_chunks(consolidated, path)
    assert read_chunks(path) == consolidated


def test_parse_all_real_data():
    """Committed EUR-Lex files: all articles in both languages, recitals from the originals."""
    chunks = parse_all()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    for regulation, n_articles, n_recitals in [("AI_ACT", 113, 180), ("GDPR", 99, 173)]:
        for lang in ("de", "en"):
            sel = [c for c in chunks if c.regulation == regulation and c.language == lang]
            articles = {c.number for c in sel if c.type == "article"}
            assert {str(i) for i in range(1, n_articles + 1)} <= articles
            assert len({c.number for c in sel if c.type == "recital"}) == n_recitals
    gdpr_en_art_6 = [c for c in chunks if c.id.startswith("GDPR-en-article-6-")]
    assert [c.paragraph for c in gdpr_en_art_6] == ["1", "2", "3", "4"]
    assert all(
        c.celex == "32016R0679" for c in chunks if c.regulation == "GDPR" and c.type == "recital"
    )
