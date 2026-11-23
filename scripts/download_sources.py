"""Download the EU AI Act and GDPR from EUR-Lex (DE + EN) into data/raw/.

Usage:
    uv run python scripts/download_sources.py            # download + write sources.json
    uv run python scripts/download_sources.py --check    # verify the committed files

EUR-Lex's HTML is WAF-protected, so the text is fetched from the Publications Office
Cellar content service (official access route); sources.json links to EUR-Lex for users.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
BASE = "https://eur-lex.europa.eu/legal-content"
DELAY_S = 2.0

# regulation -> (base CELEX, consolidated CELEX prefix, last article, title marker per language)
REGULATIONS = {
    "AI_ACT": {
        "base": "32024R1689",
        "cons": "02024R1689",
        "last": 113,
        "file": "ai_act",
        "oj_date": "2024-07-12",
    },
    "GDPR": {
        "base": "32016R0679",
        "cons": "02016R0679",
        "last": 99,
        "file": "gdpr",
        "oj_date": "2016-05-04",
    },
}
ARTICLE = {"en": "Article", "de": "Artikel"}
LANGS = ("de", "en")


class Blocked(RuntimeError):
    pass


CELLAR = "https://publications.europa.eu/resource/celex"
SPARQL = "https://publications.europa.eu/webapi/rdf/sparql"
ACCEPT_LANG = {"en": "eng", "de": "deu"}


def fetch(url: str, lang: str) -> str:
    """Fetch XHTML from the Publications Office Cellar content service."""
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": UA,
            "Accept": "application/xhtml+xml;q=0.9, text/html;q=0.8",
            "Accept-Language": ACCEPT_LANG[lang],
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="replace")
        status = resp.status
    if status != 200 or ARTICLE[lang] not in body:
        raise Blocked(f"{url} -> HTTP {status}, {len(body)} bytes (no regulation text)")
    return body


def latest_consolidated(reg: dict) -> str | None:
    """Return the newest consolidated CELEX (0YYYYRNNNN-YYYYMMDD) known to Cellar."""
    query = (
        "PREFIX cdm: <http://publications.europa.eu/ontology/cdm#> "
        "SELECT ?c WHERE { ?w cdm:resource_legal_id_celex ?c . "
        f'FILTER(STRSTARTS(STR(?c),"{reg["cons"]}-")) }}'
    )
    url = SPARQL + "?" + urllib.parse.urlencode({"query": query})
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/csv"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        csv = resp.read().decode("utf-8")
    dates = re.findall(rf"{reg['cons']}-(\d{{8}})", csv)
    return max(dates) if dates else None


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sources, failed = [], []
    for name, reg in REGULATIONS.items():
        date = latest_consolidated(reg)
        if date:
            celex, kind = f"{reg['cons']}-{date}", "consolidated"
            version = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            celex, kind, version = reg["base"], "original", reg["oj_date"]
        for lang in LANGS:
            cellar_url = f"{CELLAR}/{celex}"
            fname = f"{reg['file']}_{lang}.html"
            try:
                (RAW / fname).write_text(fetch(cellar_url, lang), encoding="utf-8")
            except Blocked as exc:
                failed.append(str(exc))
            else:
                sources.append(
                    {
                        "regulation": name,
                        "language": lang,
                        "celex": celex,
                        "version_date": version,
                        "url": f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/?uri=CELEX:{celex}",
                        "file": fname,
                        "kind": kind,
                        "fetched_from": cellar_url,
                    }
                )
            time.sleep(DELAY_S)
    if failed:
        print("Download failed for:", *failed, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    (RAW / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sources)} files to {RAW}")


def check() -> int:
    problems = []
    for reg in REGULATIONS.values():
        for lang in LANGS:
            path = RAW / f"{reg['file']}_{lang}.html"
            if not path.exists():
                problems.append(f"missing: {path.name}")
                continue
            raw = path.read_text(encoding="utf-8", errors="replace").replace("&nbsp;", " ")
            text = re.sub(r"[\s\u00a0]+", " ", raw)
            word = ARTICLE[lang]
            for n in (1, reg["last"]):
                if not re.search(rf"{word} {n}\b", text):
                    problems.append(f"{path.name}: no '{word} {n}'")
            print(f"ok  {path.name}  {len(text):>9} chars")
    sources = RAW / "sources.json"
    if not sources.exists():
        problems.append("missing: sources.json")
    else:
        entries = json.loads(sources.read_text(encoding="utf-8"))
        if len(entries) != 4 or any(not e.get("version_date") for e in entries):
            problems.append("sources.json: need 4 entries with version_date")
    for p in problems:
        print("FAIL", p, file=sys.stderr)
    print("CHECK PASSED" if not problems else f"CHECK FAILED ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify files in data/raw/")
    args = parser.parse_args()
    sys.exit(check()) if args.check else download()
