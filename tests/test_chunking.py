from assistant.chunking import naive_chunks, naive_pieces
from assistant.models import Chunk


def chunk(number, paragraph, text, reg="GDPR", lang="en", typ="article"):
    heading = f"Article {number} – Title {number}"
    return Chunk(
        id=f"{reg}-{lang}-{typ}-{number}-{paragraph}",
        text=f"{heading}\n{text}",
        regulation=reg,
        language=lang,
        type=typ,
        number=number,
        paragraph=paragraph,
        title=f"Title {number}",
        source_url=f"https://eur-lex.europa.eu/{number}",
        celex="32016R0679",
        version_date="2016-05-04",
    )


CHUNKS = [
    chunk("1", "1", "a" * 300),
    chunk("1", "2", "b" * 300),
    chunk("2", "1", "c" * 300),
    chunk("2", "2", "d" * 300),
]


def test_naive_chunks_size_and_overlap():
    pieces = naive_chunks(CHUNKS, size=200, overlap=50)
    full = "".join(p.text if i == 0 else p.text[50:] for i, p in enumerate(pieces))
    assert all(len(p.text) <= 200 for p in pieces)
    assert all(len(p.text) == 200 for p in pieces[:-1])
    for prev, nxt in zip(pieces, pieces[1:], strict=False):
        assert prev.text[-50:] == nxt.text[:50]  # consecutive pieces share 50 characters
    # Nothing is lost: undoing the overlap gives back the whole concatenated text.
    assert full.count("a") == 300 and full.count("d") == 300


def test_naive_chunks_keep_metadata_of_the_chunk_where_they_start():
    pieces = naive_chunks(CHUNKS, size=200, overlap=50)
    assert pieces[0].number == "1" and pieces[0].paragraph == "1"
    assert pieces[0].source_url == "https://eur-lex.europa.eu/1"
    last = pieces[-1]
    assert (last.number, last.paragraph) == ("2", "2")
    assert len({p.id for p in pieces}) == len(pieces)
    assert all("naive" in p.id for p in pieces)


def test_naive_chunks_write_each_heading_once_like_the_document():
    text = "".join(p.text for p in naive_chunks(CHUNKS, size=100000, overlap=0))
    assert text.count("Article 1 – Title 1") == 1
    assert text.count("Article 2 – Title 2") == 1


def test_naive_chunks_never_mix_regulations_or_languages():
    other_regulation = chunk("1", "1", "z" * 300, reg="AI_ACT")
    other_language = chunk("1", "1", "y" * 300, lang="de")
    mixed = CHUNKS + [other_regulation, other_language]
    for p in naive_chunks(mixed, size=200, overlap=50):
        letters = set(p.text) & set("abdyz")  # letters that never occur in headings
        if "z" in letters:
            assert p.regulation == "AI_ACT" and letters == {"z"}
        if "y" in letters:
            assert p.language == "de" and letters == {"y"}


def test_naive_pieces_report_every_provision_a_piece_touches():
    pieces = naive_pieces(CHUNKS, size=400, overlap=0)
    first_chunk, covered = pieces[0]
    assert first_chunk.paragraph == "1"
    # The first 400 characters run past the end of Art. 1(1) into Art. 1(2).
    assert covered == [("GDPR", "article", "1", "1"), ("GDPR", "article", "1", "2")]
