"""Embed chunks with multilingual-e5-small and store them in a persistent Chroma index."""

import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from assistant.config import Settings, get_settings
from assistant.models import Chunk

COLLECTION = "eu_ai_gdpr"
BATCH_SIZE = 256
_METADATA_FIELDS = (
    "regulation", "language", "type", "number", "paragraph",
    "title", "source_url", "celex", "version_date",
)  # fmt: skip


class E5Embeddings(Embeddings):
    """Adds the prefixes e5 models were trained with: "passage: " for stored text,
    "query: " for questions. Text that already has a prefix is left alone."""

    def __init__(self, base: Embeddings) -> None:
        self.base = base

    @staticmethod
    def _prefixed(text: str, prefix: str) -> str:
        return text if text.startswith(("query: ", "passage: ")) else prefix + text

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.base.embed_documents([self._prefixed(t, "passage: ") for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.base.embed_query(self._prefixed(text, "query: "))


def make_embeddings(settings: Settings) -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    return E5Embeddings(
        HuggingFaceEmbeddings(
            model_name=settings.embed_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    )


def _metadata(chunk: Chunk, part: int) -> dict:
    # Chroma metadata cannot hold None, so a missing paragraph/title is stored as "".
    meta = {f: getattr(chunk, f) for f in _METADATA_FIELDS}
    meta = {k: ("" if v is None else v) for k, v in meta.items()}
    meta["chunk_id"] = chunk.id
    meta["part"] = part
    return meta


def chunk_to_documents(chunk: Chunk, max_chars: int = 1200, overlap: int = 150) -> list[Document]:
    """One Document per chunk, or several if the text exceeds `max_chars`.

    The e5 model reads at most 512 tokens, so long chunks are split. Every part repeats the
    heading line so that it can be understood (and retrieved) on its own.
    """
    if len(chunk.text) <= max_chars:
        return [Document(id=chunk.id, page_content=chunk.text, metadata=_metadata(chunk, 0))]
    heading, _, body = chunk.text.partition("\n")
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    return [
        Document(
            id=f"{chunk.id}#p{n}",
            page_content=f"{heading}\n{part}",
            metadata=_metadata(chunk, n),
        )
        for n, part in enumerate(splitter.split_text(body))
    ]


def _open(embeddings: Embeddings, persist_dir: Path) -> Chroma:
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_index(chunks: list[Chunk], embeddings: Embeddings, persist_dir: Path) -> Chroma:
    """Build the index from scratch. An existing collection is dropped first, so
    rebuilding never leaves duplicates behind."""
    _open(embeddings, persist_dir).delete_collection()
    index = _open(embeddings, persist_dir)
    docs = [d for chunk in chunks for d in chunk_to_documents(chunk)]
    for start in range(0, len(docs), BATCH_SIZE):
        batch = docs[start : start + BATCH_SIZE]
        index.add_documents(batch, ids=[d.id for d in batch])
        print(f"  embedded {min(start + BATCH_SIZE, len(docs))}/{len(docs)}", flush=True)
    return index


def load_index(embeddings: Embeddings, persist_dir: Path) -> Chroma:
    return _open(embeddings, persist_dir)


def main() -> None:
    from assistant.parse import parse_all, read_chunks, write_chunks

    settings = get_settings()
    if not settings.chunks_path.exists():
        write_chunks(parse_all(), settings.chunks_path)
    chunks = read_chunks(settings.chunks_path)
    start = time.perf_counter()
    index = build_index(chunks, make_embeddings(settings), settings.chroma_dir)
    count = len(index.get()["ids"])
    print(f"{len(chunks)} chunks -> {count} documents in {time.perf_counter() - start:.0f}s")


if __name__ == "__main__":
    main()
