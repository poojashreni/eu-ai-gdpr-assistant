# CV bullets (only real, measured numbers)

Pick two or three. Replace numbers only with new measured ones.

1. Built a bilingual (German/English) RAG assistant over the EU AI Act and GDPR with LangChain, Chroma and BM25 hybrid retrieval, answering with article-level citations (1,522 provisions - articles/paragraphs, recitals, annex points - in German and English, 3,044 chunks) and instructed to refuse out-of-scope questions (refusal accuracy to be measured).

2. Measured retrieval quality on a 40-question self-written, reviewed gold set: structure-aware chunking reached 82.9% hit@6 at article level vs 54.3% for fixed-size chunking (n=35 in-scope questions; 64.3% vs 32.1% at paragraph level, n=28), which showed why splitting along legal structure matters.

3. **[Use only after the Space is live and CI has run green once]** Shipped it as a production-style service: FastAPI backend with per-IP rate limiting, a daily cap and answer caching to protect a free-tier LLM quota; Streamlit UI; 134 automated tests (no real LLM calls); Dockerised with CPU-only PyTorch and a GitHub Actions pipeline (lint, tests, eval smoke test, deploy to Hugging Face Spaces).

4. **[Add after the full evaluation with an API key]** Answer quality: citation correctness X%, refusal accuracy Y%, faithfulness Z/5 (LLM judge) on the same gold set. Do not use this bullet until the numbers exist in `eval/results.md`.

Notes for interviews: the gold set was written by you, so say "indicative" not "proven"; e5 can prefer same-language passages; the consolidated AI Act text is as of 2026-07-27.
