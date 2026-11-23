"""Naive fixed-size chunking: the baseline that structure-aware chunking is compared against.

The structured chunks (one per article paragraph / recital / annex point) are glued back into
one long text per regulation and language, in document order (recitals, articles, annexes),
and that text is cut every `size - overlap` characters, ignoring sentences and articles.
Each structured chunk repeats its article heading; in the glued text a heading is written
only once, where the provision starts, as in the real document.
"""

from itertools import groupby

from assistant.models import Chunk

_TYPE_ORDER = {"recital": 0, "article": 1, "annex": 2}


def _documents(chunks: list[Chunk]) -> list[list[Chunk]]:
    """Group chunks per (regulation, language), in document order."""
    groups: dict[tuple[str, str], list[Chunk]] = {}
    for c in chunks:
        groups.setdefault((c.regulation, c.language), []).append(c)
    return [sorted(g, key=lambda c: _TYPE_ORDER[c.type]) for g in groups.values()]


def _glue(doc: list[Chunk]) -> tuple[str, list[tuple[int, int, Chunk]]]:
    """Concatenate a document's chunks; return the text and each chunk's (start, end) span."""
    parts: list[str] = []
    spans: list[tuple[int, int, Chunk]] = []
    pos = 0
    for _, same_provision in groupby(doc, key=lambda c: (c.type, c.number)):
        for i, c in enumerate(same_provision):
            text = c.text if i == 0 else c.text.partition("\n")[2]  # heading only once
            text += "\n"
            parts.append(text)
            spans.append((pos, pos + len(text), c))
            pos += len(text)
    return "".join(parts), spans


def naive_pieces(
    chunks: list[Chunk], size: int = 1000, overlap: int = 150
) -> list[tuple[Chunk, list[tuple]]]:
    """Like `naive_chunks`, but each piece comes with the keys of every provision it touches.

    Used by the evaluation to tell "the cited provision is right" (metadata of the start chunk)
    apart from "the right text was somewhere in the piece".
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    out: list[tuple[Chunk, list[tuple]]] = []
    for doc in _documents(chunks):
        text, spans = _glue(doc)
        start_of = 0
        n = 0
        for start in range(0, len(text), size - overlap):
            end = min(start + size, len(text))
            while spans[start_of][1] <= start:
                start_of += 1
            first = spans[start_of][2]
            covered = [c.key for s, e, c in spans[start_of:] if s < end]
            piece = first.model_copy(
                update={
                    "id": f"{first.regulation}-{first.language}-naive-{n:05d}",
                    "text": text[start:end],
                }
            )
            out.append((piece, covered))
            n += 1
            if end == len(text):
                break
    return out


def naive_chunks(chunks: list[Chunk], size: int = 1000, overlap: int = 150) -> list[Chunk]:
    """Cut each regulation/language into pieces of `size` characters, `overlap` shared between
    neighbours. Each piece keeps the metadata of the chunk in which it starts."""
    return [piece for piece, _ in naive_pieces(chunks, size, overlap)]
