"""Turn the EUR-Lex HTML files into structured `Chunk`s.

One chunk = one article paragraph, one recital, or one annex point. Each chunk carries
metadata (regulation, article number, language, link) so an answer can cite its source.

Two kinds of Cellar XHTML files are involved (see `data/raw/sources.json`):

* role "articles"  - consolidated text: articles and annexes (`title-article-norm`, `norm`,
  `no-parag`, `grid-list`, `title-annex-*`, amendment markers `modref`). It has no recitals.
* role "recitals"  - the original Official Journal text: recitals live in `div#rct_N` tables.
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from assistant.models import Chunk

# Words that make a chunk understandable on its own, per language.
WORDS = {
    "en": {"article": "Article", "recital": "Recital", "annex": "Annex", "point": "point"},
    "de": {"article": "Artikel", "recital": "Erwägungsgrund", "annex": "Anhang", "point": "Punkt"},
}

INLINE_TAGS = {"span", "a", "i", "b", "em", "strong", "sup", "sub", "u"}
# "1." / "(1)" / "1a." / "(44a)" -> "1" / "1" / "1a" / "44a"
_PARA_MARKER = re.compile(r"^\(?(\d+[a-z]*)[.)]\s*")
# Top-level annex point: "3. Quality ..." but not the sub-point "3.1. ..."
_ANNEX_POINT = re.compile(r"^(\d+)\.\s")
_SECTION = re.compile(r"^(?:Section|Abschnitt)\s+([A-Z0-9]+)\b")


# --------------------------------------------------------------------------- text helpers


def _render(node) -> str:
    """HTML node -> text. Inline tags join directly, block tags get their own line."""
    if isinstance(node, NavigableString):
        return str(node)
    if node.name in INLINE_TAGS:
        return "".join(_render(c) for c in node.children)
    if node.name == "br":
        return "\n"
    if node.name == "tr":
        cells = [_clean(_render(td)).replace("\n", " ") for td in node.find_all(["td", "th"])]
        return "\n" + " ".join(cells) + "\n"
    if "grid-container" in (node.get("class") or []):
        # A list item: marker column ("(a)") and text column side by side.
        cols = [_clean(_render(c)) for c in node.find_all(recursive=False)]
        return "\n" + " ".join(cols) + "\n"
    return "\n" + "".join(_render(c) for c in node.children) + "\n"


def _clean(text: str) -> str:
    """Normalise whitespace (incl. nbsp); keep one line per block."""
    lines = (re.sub(r"[ \t\r\f\v\xa0]+", " ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def _text(node) -> str:
    return _clean(_render(node))


def _strip_noise(soup: BeautifulSoup) -> None:
    """Remove amendment markers (▼M1, ▼B ...) and footnote reference links like "(1)"."""
    for marker in soup.select("p.modref"):
        marker.decompose()
    for tag in soup.select("a:has(> span.oj-note-tag)"):
        previous = tag.previous_sibling
        following = tag.next_sibling
        # "market (1)." -> "market."  (but keep the space in "market (1) and")
        punct_follows = isinstance(following, NavigableString) and following.lstrip()[:1] in ",.;:)"
        if isinstance(previous, NavigableString) and punct_follows:
            previous.replace_with(previous.rstrip())
        tag.decompose()


def _anchor(source_url: str, element_id: str | None) -> str:
    return f"{source_url}#{element_id}" if element_id else source_url


def _make_chunk(*, ctype, number, paragraph, title, text, element_id, **meta) -> Chunk:
    parts = [meta["regulation"], meta["language"], ctype, number]
    if paragraph:
        parts.append(paragraph)
    return Chunk(
        id="-".join(parts),
        text=text,
        type=ctype,
        number=number,
        paragraph=paragraph,
        title=title,
        regulation=meta["regulation"],
        language=meta["language"],
        celex=meta["celex"],
        version_date=meta["version_date"],
        source_url=_anchor(meta["source_url"], element_id),
    )


# --------------------------------------------------------------------------- articles


def _article_units(article: Tag) -> list[tuple[str | None, str]]:
    """Split an article into (paragraph number, text) units.

    A unit starts at a numbered paragraph (`div.norm` with a `no-parag` span) or a numbered
    definition (`grid-list` item such as "(1)" in Article 3). Unnumbered blocks before the first
    unit are a lead-in ("For the purposes of this Regulation:") and are repeated in each unit;
    unnumbered blocks after a unit belong to it.
    """
    lead: list[str] = []
    units: list[tuple[str, list[str]]] = []
    for child in article.find_all(recursive=False):
        classes = child.get("class") or []
        if "title-article-norm" in classes or "eli-title" in classes:
            continue
        number, body = None, None
        marker = child.find("span", class_="no-parag", recursive=False)
        if marker is not None:
            number = _PARA_MARKER.match(_clean(marker.get_text()) + " ")
            marker.decompose()
            body = _text(child)
        elif "grid-container" in classes:
            text = _text(child)
            number = _PARA_MARKER.match(text)
            body = text[number.end() :] if number else text
        text = body if body is not None else _text(child)
        if not text:
            continue
        if number:
            units.append((number.group(1), [text]))
        elif units:
            units[-1][1].append(text)
        else:
            lead.append(text)

    if not units:
        return [(None, "\n".join(lead))] if lead else []
    return [(n, "\n".join([*lead, f"({n}) {blocks[0]}", *blocks[1:]])) for n, blocks in units]


def _parse_articles(soup: BeautifulSoup, **meta) -> list[Chunk]:
    words = WORDS[meta["language"]]
    chunks = []
    for article in soup.select("div.eli-subdivision[id^=art_]"):
        element_id = article["id"]
        number = element_id.removeprefix("art_")
        title_el = article.select_one(".stitle-article-norm")
        title = _text(title_el).rstrip("'") if title_el else None
        heading = f"{words['article']} {number}" + (f" – {title}" if title else "")
        for paragraph, text in _article_units(article):
            chunks.append(
                _make_chunk(
                    ctype="article",
                    number=number,
                    paragraph=paragraph,
                    title=title,
                    text=f"{heading}\n{text}",
                    element_id=element_id,
                    **meta,
                )
            )
    return chunks


# --------------------------------------------------------------------------- annexes


def _annex_units(annex: Tag) -> tuple[list[str], list[tuple[str | None, str | None, list[str]]]]:
    """Return (intro blocks, [(section, point, blocks)]) for one annex.

    Annex markup is irregular, so points are recognised by their number ("3. ...") whether the
    block is a list item, a heading or a plain paragraph. "Section A" style headings are kept
    with the next point (and tell apart points whose numbering restarts in each section).
    """
    intro: list[str] = []
    points: list[tuple[str | None, str | None, list[str]]] = []
    pending: list[str] = []
    section = None
    current = None
    titles_seen = 0
    for child in annex.find_all(recursive=False):
        classes = child.get("class") or []
        if any(c.startswith("title-annex") for c in classes):
            titles_seen += 1
            continue
        text = _text(child)
        if not text or child.name == "hr":
            continue
        section_match = None
        if any(c.startswith("title-gr-seq") for c in classes):
            section_match = _SECTION.match(text)
        if section_match:
            section, current, pending = section_match.group(1), None, [text]
            continue
        number = _ANNEX_POINT.match(text)
        if number:
            current = (section, number.group(1), [*pending, text[number.end() :]])
            points.append(current)
            pending = []
        elif current is not None:
            current[2].append(text)
        elif section is not None:
            pending.append(text)
        else:
            intro.append(text)
    return intro, points


def _parse_annexes(soup: BeautifulSoup, **meta) -> list[Chunk]:
    words = WORDS[meta["language"]]
    chunks = []
    for annex in soup.find_all("div", id=re.compile(r"^anx_")):
        element_id = annex["id"]
        numeral = element_id.removeprefix("anx_")
        titles = annex.find_all(class_=re.compile(r"^title-annex"), recursive=False)
        title = _text(titles[1]) if len(titles) > 1 else None
        heading = f"{words['annex']} {numeral}" + (f" – {title}" if title else "")
        intro, points = _annex_units(annex)

        numbers = [n for _, n, _ in points]
        restarts = len(numbers) != len(set(numbers))  # e.g. Section A and B both have a point 1
        units: list[tuple[str | None, str]] = []
        if intro or not points:
            units.append((None, heading + ("\n" + "\n".join(intro) if intro else "")))
        for section, number, blocks in points:
            paragraph = f"{section}.{number}" if restarts and section else number
            units.append(
                (paragraph, f"{heading}, {words['point']} {paragraph}\n" + "\n".join(blocks))
            )
        for paragraph, text in units:
            chunks.append(
                _make_chunk(
                    ctype="annex",
                    number=numeral,
                    paragraph=paragraph,
                    title=title,
                    text=text,
                    element_id=element_id,
                    **meta,
                )
            )
    return chunks


# --------------------------------------------------------------------------- recitals


def _parse_recitals(soup: BeautifulSoup, **meta) -> list[Chunk]:
    word = WORDS[meta["language"]]["recital"]
    chunks = []
    for recital in soup.select("div.eli-subdivision[id^=rct_]"):
        element_id = recital["id"]
        number = element_id.removeprefix("rct_")
        cells = recital.select("tr > td")
        # First cell is the "(26)" label, the second one holds the text.
        body = _text(cells[1]) if len(cells) > 1 else _text(recital)
        chunks.append(
            _make_chunk(
                ctype="recital",
                number=number,
                paragraph=None,
                title=None,
                text=f"{word} {number}\n{body}",
                element_id=element_id,
                **meta,
            )
        )
    return chunks


# --------------------------------------------------------------------------- public API


def parse_regulation(
    html: str,
    *,
    regulation: str,
    language: str,
    source_url: str,
    celex: str,
    version_date: str,
    role: str = "articles",
) -> list[Chunk]:
    """Parse one EUR-Lex HTML file.

    role="articles" extracts articles and annexes (consolidated text);
    role="recitals" extracts recitals (original Official Journal text).
    """
    if role not in ("articles", "recitals"):
        raise ValueError(f"unknown role: {role!r}")
    html = re.sub(r"^\s*<\?xml[^>]*\?>", "", html)  # real files start with an XML declaration
    soup = BeautifulSoup(html, "lxml")
    _strip_noise(soup)
    meta = dict(
        regulation=regulation,
        language=language,
        source_url=source_url,
        celex=celex,
        version_date=version_date,
    )
    if role == "recitals":
        return _parse_recitals(soup, **meta)
    return _parse_articles(soup, **meta) + _parse_annexes(soup, **meta)


def parse_all(raw_dir: Path = Path("data/raw")) -> list[Chunk]:
    """Parse every file listed in `sources.json` according to its role."""
    sources = json.loads((raw_dir / "sources.json").read_text(encoding="utf-8"))
    chunks: list[Chunk] = []
    for entry in sources:
        html = (raw_dir / entry["file"]).read_text(encoding="utf-8")
        chunks += parse_regulation(
            html,
            regulation=entry["regulation"],
            language=entry["language"],
            source_url=entry["url"],
            celex=entry["celex"],
            version_date=entry["version_date"],
            role=entry["role"],
        )
    return chunks


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    with path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


if __name__ == "__main__":
    from collections import Counter

    from assistant.config import get_settings

    all_chunks = parse_all()
    out = get_settings().chunks_path
    write_chunks(all_chunks, out)
    print(f"Wrote {len(all_chunks)} chunks to {out}")
    counts = Counter((c.regulation, c.language, c.type) for c in all_chunks)
    for (regulation, language, ctype), n in sorted(counts.items()):
        print(f"  {regulation:7} {language}  {ctype:8} {n}")
