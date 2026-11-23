"""Hybrid retrieval: BM25 (keywords) + Chroma (meaning), fused, then de-duplicated."""

import re
from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from assistant.config import Settings
from assistant.index import chunk_to_documents
from assistant.models import Chunk

_WORD = re.compile(r"\w+")


def bm25_tokens(text: str) -> list[str]:
    """Lowercase and split on non-word characters (Unicode-aware, so umlauts and ß stay).

    "Art. 6(2)" -> ["art", "6", "2"]; "Artikel 5" matches the heading "Artikel 5 - ...".
    No stemming in v1.
    """
    return _WORD.findall(text.lower())


def _key(doc: Document) -> tuple:
    m = doc.metadata
    return (m["regulation"], m["type"], m["number"], m.get("paragraph", ""))


def dedupe_cross_language(
    docs: list[Document], k: int, prefer_language: str | None
) -> list[Document]:
    """Reduce a ranked list to at most `k` documents, one per legal provision.

    A provision is identified by (regulation, type, number, paragraph) from the metadata, with
    paragraph "" meaning "none". The parts of a split chunk and the DE/EN copies of the same
    provision therefore share a key and collapse into a single result.

    The result keeps the rank position of the first occurrence of each key. The document shown
    in that slot is the first one of the key, unless `prefer_language` is set, that document is
    in another language, and a document of the preferred language with the same key appears
    anywhere in `docs` -- then the preferred-language one is returned instead.
    """
    slots: dict[tuple, Document] = {}
    for d in docs:
        key = _key(d)
        if key not in slots:
            slots[key] = d
        elif (
            prefer_language
            and slots[key].metadata.get("language") != prefer_language
            and d.metadata.get("language") == prefer_language
        ):
            slots[key] = d
    return list(slots.values())[:k]


def build_retriever(
    vectorstore: Chroma, chunks: list[Chunk], settings: Settings
) -> Callable[[str, str | None], list[Document]]:
    """Return `retrieve(question, language)`. BM25 indexes the same Documents as Chroma."""
    fetch = 2 * settings.top_k
    docs = [d for c in chunks for d in chunk_to_documents(c)]
    bm25 = BM25Retriever.from_documents(docs, k=fetch, preprocess_func=bm25_tokens)
    # The store embeds the query itself (E5Embeddings adds "query: "), so no prefix here.
    vector = vectorstore.as_retriever(search_kwargs={"k": fetch})
    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector], weights=[settings.bm25_weight, settings.vector_weight]
    )

    def retrieve(question: str, language: str | None = None) -> list[Document]:
        return dedupe_cross_language(ensemble.invoke(question), settings.top_k, language)

    return retrieve
