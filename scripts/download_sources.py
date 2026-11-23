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
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DELAY_S = 2.0

# Per regulation: CELEX ids, last article / recital number, file prefix, OJ publication date.
REGULATIONS = {
    "AI_ACT": {
        "base": "32024R1689",
        "cons": "02024R1689",
        "last": 113,
        "last_recital": 180,
        "file": "ai_act",
        "oj_date": "2024-07-12",
    },
    "GDPR": {
        "base": "32016R0679",
        "cons": "02016R0679",
        "last": 99,
        "last_recital": 173,
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


def _entry(name, reg, lang, celex, kind, version, fname, role):
    cellar_url = f"{CELLAR}/{celex}"
    return cellar_url, {
        "regulation": name,
        "language": lang,
        "celex": celex,
        "version_date": version,
        "url": f"https://eur-lex.europa.eu/legal-content/{lang.upper()}/TXT/?uri=CELEX:{celex}",
        "file": fname,
        "kind": kind,
        "role": role,
        "fetched_from": cellar_url,
    }


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sources, failed = [], []
    try:
        jobs = []
        for name, reg in REGULATIONS.items():
            date = latest_consolidated(reg)
            if date:
                celex, kind = f"{reg['cons']}-{date}", "consolidated"
                version = f"{date[:4]}-{date[4:6]}-{date[6:]}"
                jobs.append((name, reg, celex, kind, version, "", "articles"))
            # Consolidated texts carry no recitals: always add the original OJ text.
            jobs.append(
                (name, reg, reg["base"], "original", reg["oj_date"], "_original", "recitals")
            )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        print(f"Download failed: SPARQL lookup: {exc}", file=sys.stderr)
        sys.exit(1)
    for name, reg, celex, kind, version, suffix, role in jobs:
        for lang in LANGS:
            fname = f"{reg['file']}_{lang}{suffix}.html"
            cellar_url, entry = _entry(name, reg, lang, celex, kind, version, fname, role)
            try:
                (RAW / fname).write_text(fetch(cellar_url, lang), encoding="utf-8")
            except (Blocked, urllib.error.URLError, TimeoutError, OSError) as exc:
                failed.append(f"{cellar_url} ({lang}): {exc}")
            else:
                sources.append(entry)
            time.sleep(DELAY_S)
    if failed:
        print("Download failed for:", *failed, sep="\n  ", file=sys.stderr)
        sys.exit(1)
    (RAW / "sources.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(sources)} files to {RAW}")


def _normalized(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="replace").replace("&nbsp;", " ")
    return re.sub(r"[\s\u00a0]+", " ", raw)


def check() -> int:
    problems = []
    for reg in REGULATIONS.values():
        for lang in LANGS:
            path = RAW / f"{reg['file']}_{lang}.html"
            if not path.exists():
                problems.append(f"missing: {path.name}")
            else:
                text = _normalized(path)
                word = ARTICLE[lang]
                for n in (1, reg["last"]):
                    if not re.search(rf"{word} {n}\b", text):
                        problems.append(f"{path.name}: no '{word} {n}'")
                print(f"ok  {path.name}  {len(text):>9} chars")
            orig = RAW / f"{reg['file']}_{lang}_original.html"
            if not orig.exists():
                problems.append(f"missing: {orig.name}")
                continue
            text = _normalized(orig)
            for n in (1, reg["last_recital"]):
                if not re.search(rf"\({n}\)", text):
                    problems.append(f"{orig.name}: no recital ({n})")
            print(f"ok  {orig.name}  {len(text):>9} chars")
    sources = RAW / "sources.json"
    if not sources.exists():
        problems.append("missing: sources.json")
    else:
        entries = json.loads(sources.read_text(encoding="utf-8"))
        if len(entries) != 8 or any(
            not e.get("version_date") or not e.get("role") for e in entries
        ):
            problems.append("sources.json: need 8 entries with version_date and role")
    for p in problems:
        print("FAIL", p, file=sys.stderr)
    print("CHECK PASSED" if not problems else f"CHECK FAILED ({len(problems)} problems)")
    return 1 if problems else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify files in data/raw/")
    args = parser.parse_args()
    sys.exit(check()) if args.check else download()
