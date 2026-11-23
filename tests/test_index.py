from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings

from assistant.index import (
    COLLECTION,
    E5Embeddings,
    build_index,
    chunk_to_documents,
    load_index,
)
from assistant.models import Chunk


class RecordingEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.docs: list[str] = []
        self.queries: list[str] = []

    def embed_documents(self, texts):
        self.docs.extend(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        self.queries.append(text)
        return [1.0, 0.0]


def make_chunk(id="GDPR-en-article-6-1", text="Article 6 – Lawfulness\n(1) Body.", **kw):
    base = dict(
        id=id,
        text=text,
        regulation="GDPR",
        language="en",
        type="article",
        number="6",
        paragraph="1",
        title="Lawfulness",
        source_url="https://eur-lex.europa.eu/x#art_6",
        celex="32016R0679",
        version_date="2016-04-27",
    )
    base.update(kw)
    return Chunk(**base)


def test_e5_prefixes():
    base = RecordingEmbeddings()
    e5 = E5Embeddings(base)
    e5.embed_documents(["hello", "passage: already"])
    e5.embed_query("what?")
    e5.embed_query("query: already")
    assert base.docs == ["passage: hello", "passage: already"]
    assert base.queries == ["query: what?", "query: already"]


def test_short_chunk_single_document():
    docs = chunk_to_documents(make_chunk())
    assert len(docs) == 1
    d = docs[0]
    assert d.id == "GDPR-en-article-6-1"
    assert d.page_content.startswith("Article 6")
    assert d.metadata["chunk_id"] == d.id
    assert d.metadata["part"] == 0
    assert d.metadata["regulation"] == "GDPR"
    assert d.metadata["paragraph"] == "1"


def test_none_metadata_stored_as_empty_string():
    docs = chunk_to_documents(make_chunk(paragraph=None, title=None))
    assert docs[0].metadata["paragraph"] == ""
    assert docs[0].metadata["title"] == ""
    assert all(v is not None for v in docs[0].metadata.values())


def test_long_chunk_split_with_heading():
    heading = "Annex III – High-risk"
    body = " ".join(f"Sentence number {i} about biometrics." for i in range(200))
    chunk = make_chunk(id="AI_ACT-en-annex-III-1", text=f"{heading}\n{body}")
    docs = chunk_to_documents(chunk, max_chars=300, overlap=30)
    assert len(docs) > 1
    for n, d in enumerate(docs):
        assert d.id == f"AI_ACT-en-annex-III-1#p{n}"
        assert d.page_content.startswith(heading + "\n")
        assert len(d.page_content) <= 300 + len(heading) + 1
        assert d.metadata["part"] == n
        assert d.metadata["chunk_id"] == "AI_ACT-en-annex-III-1"


def test_metadata_roundtrip(tmp_path):
    index = build_index([make_chunk()], DeterministicFakeEmbedding(size=8), tmp_path)
    hit = index.similarity_search("anything", k=1)[0]
    assert hit.metadata["regulation"] == "GDPR"
    assert hit.metadata["number"] == "6"
    assert hit.metadata["source_url"].startswith("https://eur-lex")


def test_load_index_reads_persisted(tmp_path):
    emb = DeterministicFakeEmbedding(size=8)
    chunks = [make_chunk(), make_chunk(id="GDPR-en-article-6-2", paragraph="2")]
    build_index(chunks, emb, tmp_path)
    loaded = load_index(emb, tmp_path)
    assert len(loaded.get()["ids"]) == 2
    assert loaded._collection.name == COLLECTION


def test_rebuild_does_not_duplicate(tmp_path):
    emb = DeterministicFakeEmbedding(size=8)
    chunks = [make_chunk(), make_chunk(id="GDPR-en-article-6-2", paragraph="2")]
    build_index(chunks, emb, tmp_path)
    index = build_index(chunks, emb, tmp_path)
    assert len(index.get()["ids"]) == 2
