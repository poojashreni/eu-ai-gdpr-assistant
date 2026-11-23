from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from assistant.config import Settings
from assistant.index import build_index
from assistant.models import Chunk
from assistant.retrieve import bm25_tokens, build_retriever, dedupe_cross_language


def doc(reg="AI_ACT", typ="article", number="6", paragraph="2", lang="en", part=0, text="x"):
    return Document(
        page_content=f"{text} {lang} {part}",
        metadata={
            "regulation": reg, "type": typ, "number": number, "paragraph": paragraph,
            "language": lang, "part": part,
        },
    )  # fmt: skip


def chunk(number, lang, text, reg="AI_ACT", paragraph=None):
    return Chunk(
        id=f"{reg}-{lang}-article-{number}-{paragraph or ''}",
        text=text,
        regulation=reg,
        language=lang,
        type="article",
        number=number,
        paragraph=paragraph,
        title=None,
        source_url="https://eur-lex.europa.eu/x",
        celex="32024R1689",
        version_date="2024-06-13",
    )


def test_tokenizer_keeps_umlauts_and_numbers():
    assert bm25_tokens("Artikel 5 der KI-Verordnung, Größe!") == [
        "artikel", "5", "der", "ki", "verordnung", "größe",
    ]  # fmt: skip
    assert bm25_tokens("Art. 6(2)") == ["art", "6", "2"]


def test_dedupe_merges_languages():
    docs = [doc(lang="en"), doc(lang="de"), doc(number="7", lang="en")]
    out = dedupe_cross_language(docs, k=5, prefer_language="de")
    assert [(d.metadata["number"], d.metadata["language"]) for d in out] == [
        ("6", "de"),
        ("7", "en"),
    ]
    out = dedupe_cross_language(docs, k=5, prefer_language="en")
    assert [d.metadata["language"] for d in out] == ["en", "en"]


def test_dedupe_without_preference_keeps_first():
    docs = [doc(lang="en"), doc(lang="de")]
    out = dedupe_cross_language(docs, k=5, prefer_language=None)
    assert [d.metadata["language"] for d in out] == ["en"]


def test_dedupe_merges_parts():
    docs = [doc(part=0), doc(part=1), doc(part=2), doc(number="9")]
    out = dedupe_cross_language(docs, k=5, prefer_language="en")
    assert [d.metadata["number"] for d in out] == ["6", "9"]
    assert out[0].metadata["part"] == 0  # highest-ranked part wins


def test_dedupe_empty_paragraph_is_a_key():
    docs = [doc(paragraph="", lang="en"), doc(paragraph="", lang="de"), doc(paragraph="1")]
    assert len(dedupe_cross_language(docs, k=5, prefer_language="de")) == 2


def test_dedupe_respects_k():
    docs = [doc(number=str(n)) for n in range(10)]
    assert len(dedupe_cross_language(docs, k=3, prefer_language=None)) == 3


def test_article_number_query_hits_bm25(tmp_path):
    chunks = [
        chunk("5", "de", "Artikel 5 – Verbotene Praktiken\nFolgende Praktiken sind verboten."),
        chunk(
            "5", "en", "Article 5 – Prohibited practices\nThe following practices are prohibited."
        ),
        chunk("6", "de", "Artikel 6 – Einstufungsvorschriften\nEin KI-System gilt als Hochrisiko."),
        chunk("9", "de", "Artikel 9 – Risikomanagementsystem\nEin System wird eingerichtet."),
        chunk("10", "de", "Artikel 10 – Daten\nTrainingsdaten müssen geeignet sein."),
        chunk("4", "de", "Artikel 4 – KI-Kompetenz\nAnbieter fördern Kompetenz."),
    ]
    store = build_index(chunks, DeterministicFakeEmbedding(size=16), tmp_path)
    settings = Settings(top_k=3)
    retrieve = build_retriever(store, chunks, settings)
    results = retrieve("Was regelt Artikel 5 der KI-Verordnung?", "de")
    assert len(results) <= 3
    assert ("AI_ACT", "5") in [(d.metadata["regulation"], d.metadata["number"]) for d in results]
    assert results[0].metadata["language"] == "de"
