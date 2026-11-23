"""Download the EU AI Act and GDPR from EUR-Lex (DE + EN) into data/raw/.

Usage:
    uv run python scripts/download_sources.py            # download + write sources.json
    uv run python scripts/download_sources.py --check    # verify the committed files

If EUR-Lex answers with its bot challenge (empty 202 / JavaScript page), the script
stops with the list of failed URLs. It does not try to bypass the challenge: save the
pages manually from a browser under the file names below and run --check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
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
    "AI_ACT": {"base": "32024R1689", "cons": "02024R1689", "last": 113, "file": "ai_act"},
    "GDPR": {"base": "32016R0679", "cons": "02016R0679", "last": 99, "file": "gdpr"},
}
ARTICLE = {"en": "Article", "de": "Artikel"}
LANGS = ("de", "en")


class Blocked(RuntimeError):
    pass


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Language": "en,de"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        body = resp.read().decode("utf-8", errors="replace")
        status = resp.status
    if status != 200 or "Article" not in body and "Artikel" not in body:
        raise Blocked(f"{url} -> HTTP {status}, {len(body)} bytes (bot challenge?)")
    return body


def latest_consolidated(reg: dict) -> str | None:
    """Return YYYYMMDD of the newest consolidated version listed on the ALL page."""
    html = fetch(f"{BASE}/EN/ALL/?uri=CELEX:{reg['base']}")
    dates = re.findall(rf"{reg['cons']}-(\d{{8}})", html)
    return max(dates) if dates else None


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    sources, failed = [], []
    for name, reg in REGULATIONS.items():
        try:
            date = latest_consolidated(reg)
        except Blocked as exc:
            failed.append(str(exc))
            continue
        finally:
            time.sleep(DELAY_S)
        if date:
            celex, kind = f"{reg['cons']}-{date}", "consolidated"
            version = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        else:
            celex, kind, version = reg["base"], "original", ""  # fill in OJ date manually
        for lang in LANGS:
            url = f"{BASE}/{lang.upper()}/TXT/HTML/?uri=CELEX:{celex}"
            fname = f"{reg['file']}_{lang}.html"
            try:
                (RAW / fname).write_text(fetch(url), encoding="utf-8")
            except Blocked as exc:
                failed.append(str(exc))
            else:
                sources.append(
                    {"regulation": name, "language": lang, "celex": celex,
                     "version_date": version, "url": url, "file": fname, "kind": kind}
                )
            time.sleep(DELAY_S)
    if failed:
        print("EUR-Lex blocked or failed for:", *failed, sep="\n  ", file=sys.stderr)
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
            text = path.read_text(encoding="utf-8", errors="replace")
            word = ARTICLE[lang]
            for n in (1, reg["last"]):
                if not re.search(rf"{word}\s+{n}\b", text):
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
