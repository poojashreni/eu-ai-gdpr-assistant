# LEARNING.md — How to read this project

This file is your guide to what we build and why. The project is a chat assistant that answers questions about the EU AI Act and the GDPR, in German or English, and always points to the exact article it used.

We build it in eight stages. Each stage adds one idea and (from stage 1 on) comes with a notebook you can run, and a section below that explains it in plain words.

How to use it:

1. Read the section for the stage first, then open the matching notebook.
2. Run the notebook cell by cell. Change something and see what happens: that is the fastest way to learn.
3. Do not move on until the stage makes sense. Each later stage builds on the earlier ones.

The technique behind it is called RAG (retrieval-augmented generation): instead of letting a language model answer from memory, we first look up the relevant passages of the law, then ask the model to answer using only those passages. That is how the answers stay grounded and can be cited.

## §1 Load & parse

**Notebook:** `notebooks/01_load_and_parse.ipynb` | **Code:** `src/assistant/parse.py`

**What is a chunk?** A chunk is one small piece of the law that we can store, search and hand to the language model on its own. In this project one chunk is one numbered paragraph of an article, one recital, or one annex point. Example: "GDPR Article 17, paragraph 1" is one chunk (about 500 characters).

**Why split by article and paragraph?** The language model can only read a limited amount of text per question, and a search works better on a focused passage than on a 50-page law. We could cut the text every 1000 characters, but then a cut might land in the middle of a sentence and a chunk would not know which article it belongs to. The law already has a clean structure (articles, numbered paragraphs, points), so we cut along it. Every chunk is a complete legal statement, and it starts with its own heading ("Article 17 - Right to erasure") so it is understandable when read alone. (Stage 2 will compare this with plain fixed-size cutting, so you can see the difference in numbers.)

**Why metadata is what makes citations possible.** Next to its text, every chunk carries: the regulation (AI Act or GDPR), the type (article, recital, annex), the article number and paragraph, the language (German or English), and a link to EUR-Lex that jumps to the right article. The language model never has to invent a citation: when a chunk is retrieved, we already know "GDPR Art. 17(1)" and the link, and we just print them next to the answer. If we had lost that information while parsing, no later stage could bring it back. Metadata is also how we later recognise that a German and an English chunk are the same provision (same regulation, type, number and paragraph).

**Consolidated vs. original text (a decision you should understand).** EUR-Lex offers two versions of each law:

- the *original* text, exactly as published in the Official Journal in 2016 / 2024, and
- the *consolidated* text, which merges later amendments into one up-to-date version. The AI Act was amended in 2026, so the consolidated version has new articles such as 4a and 75a-75d.

An assistant should quote the law as it applies today, so **articles and annexes come from the consolidated files**. The catch: consolidated texts do not contain the *recitals* (the numbered explanatory "Whereas ..." paragraphs before Article 1). Recitals are still useful because they explain what the rules are meant to achieve, so **recitals come from the original files**, and each chunk records which version (CELEX number and date) it came from. We never take articles from the originals, otherwise every article would appear twice, once amended and once not.

**What the parser had to deal with.** Amendment markers like "M1" and footnote numbers had to be removed, article numbers like "4a" kept as text, and annexes handled one by one because each is laid out a bit differently.

**Check yourself:** (1) Why does each chunk repeat the article title? (2) Where would the word "Article 6" in a citation come from if the model had not been given metadata? (3) Why do we not read recitals from the consolidated files?

## §2 Chunking comparison

**Notebook:** `notebooks/02_chunking.ipynb` | **Code:** `src/assistant/chunking.py`

**The question.** In stage 1 we cut the law along its own structure: one chunk per article paragraph, recital or annex point. Was that worth the effort, or would the simple method most RAG tutorials use, cutting the text every N characters, work just as well? Stage 2 answers this with numbers instead of opinion.

**The naive baseline.** `naive_chunks` glues each regulation (per language) back into one long text, in document order, and cuts it into pieces of 1000 characters, each overlapping the previous one by 150 characters. The overlap is there so a sentence cut at the edge of one piece still appears whole in the next. A piece cannot know which article it belongs to, so it gets the label of the provision in which it *starts*. That gives 2406 naive pieces against 3044 structured chunks.

**What goes wrong with naive pieces.** The notebook shows a real one: it is labelled GDPR Art. 6(4), starts in the middle of a word ("cular whether ..."), and about 60 % of its text is actually Art. 7 (consent), ending mid-sentence at "(3) The d". If the assistant answers a consent question from it, the citation says "Art. 6(4)": wrong, even if the answer is right.

**How we compared.** Same embedding model, same hybrid retrieval, same settings (top_k = 6, BM25 0.4 / vectors 0.6); only the chunks differ. Then we ran the 35 in-scope questions of the gold set (stage 6) and counted how often the expected article was among the 6 results (hit@k).

| | structured | naive |
|---|---|---|
| hit@k, article | **82.9%** (29/35) | 54.3% (19/35) |
| hit@k, paragraph | **64.3%** (18/28) | 32.1% (9/28) |
| hit@k, article, counting text under a wrong label | n/a | 57.1% (20/35) |

The last row is generous to naive: it counts a hit if the expected article's text is anywhere inside a retrieved piece. Even then naive is far behind, so the gap is not only about labels: pieces that mix articles and start mid-sentence are also harder to *find*. On the 10 questions where the two methods disagreed, structured won all 10.

**What this does and does not prove.** It supports the design decision of stage 1 on our data. It does not prove structured chunking is better in general: 35 questions is small (one question is about 3 points), we wrote the questions ourselves knowing the system (they may be "article-shaped", which suits structured chunks), and we compared against the simplest naive version, not a tuned one (other sizes, cutting at sentence ends). The gold set was reviewed by the author on 2026-11-23 (no changes), so these numbers are measured on a reviewed set, but they stay indicative: n = 35, and the questions are self-written. A fair CV sentence: "structure-aware chunking reached 82.9% article-level hit@6 vs 54.3% for fixed-size chunks (n = 35, self-written questions)".

**Check yourself:** (1) Why does a naive piece get the label of the provision where it starts, and what can go wrong? (2) Why do we also report the "text under a wrong label" number? (3) Name two reasons the 83% vs 54% might not hold for real users' questions.

## §3 Embeddings + Chroma

**Notebook:** `notebooks/03_embeddings.ipynb` | **Code:** `src/assistant/index.py`

**What is an embedding?** An embedding is a list of numbers (here 384 of them) that a model computes from a text. The model is trained so that texts with a similar *meaning* get similar lists, even when they use different words or even different languages. You can think of each text as a point in a 384-dimensional space: close points mean similar meaning. Searching then becomes "embed the question, find the nearest points". We use `multilingual-e5-small`, a small model that runs on a laptop CPU and understands German and English.

**Why the "query:" and "passage:" prefixes?** The e5 models were trained with two labels: stored texts start with `passage: ` and questions start with `query: `. If you leave them out, the search still works but noticeably worse, because the model expects them. Our `E5Embeddings` wrapper adds them automatically (and never adds them twice).

**Normalization and cosine.** We scale every vector to length 1. Then the only thing left that distinguishes two vectors is their direction, and "how similar" is the cosine of the angle between them (1.0 = same direction). Chroma is set to cosine distance, where small means similar (distance = 1 - similarity). A tip from the notebook: e5 gives high scores to everything (even unrelated legal texts score around 0.8), so compare the *gap* between scores, not the absolute number.

**Why long chunks are split.** The model reads at most 512 tokens (about 1500-2000 characters); the rest would be silently ignored. Some chunks (for example AI Act Article 5(1)) are over 6000 characters, so any chunk over 1200 characters is cut into parts of up to 1200 characters with 150 characters of overlap. **Every part repeats the heading line** ("Artikel 5 - Verbotene Praktiken ..."), so a part still says which article it belongs to. The metadata is the same for all parts, plus a `part` number, so a citation never changes. Result: 3044 chunks became 3638 documents (594 extra parts).

**What Chroma stores.** For each document: its id (like `GDPR-en-article-6-1`, or `...#p1` for a split part), the text, the embedding vector, and the metadata (regulation, language, article number, paragraph, link ...). Chroma cannot store "empty" values, so a missing paragraph or title is saved as an empty string `""`. The index lives in the folder `data/chroma/` (not in git; rebuild it any time with `uv run python -m assistant.index`). Rebuilding deletes the old collection first, so you never get duplicates.

**Cost of building it (MacBook Air M1, CPU).** About 131 seconds for 3638 documents, peak memory about 1.6 GB. Fine for 8 GB RAM.

**What you will see in the notebook:** German and English versions of GDPR Art. 6(1) have similarity 0.87; an unrelated article scores 0.81 / 0.77. A German question returns only German chunks at first; filtering to English still finds related English chunks, but looser ones. That is why stage 4 adds keyword search and de-duplication.

**Check yourself:** (1) What would go wrong if we stored chunks without `passage: ` but queried with `query: `? (2) Why does each split part repeat the heading? (3) Why compare score gaps instead of absolute scores?

## §4 Hybrid retrieval

**Two kinds of search.** *Keyword search* (we use BM25) scores a passage by how many of the question's words it contains, giving more weight to rare words. *Meaning search* uses the embeddings from stage 3: it finds passages whose meaning is close to the question, even with different words. Each has a blind spot. Keywords cannot see that "anonym" and "nicht mehr identifiziert werden kann" mean the same thing. Embeddings are fuzzy about exact tokens such as "Artikel 5" or "Annex III".

**Why law needs both.** People ask in two ways: by reference ("Was regelt Artikel 5?") and by idea ("When is data anonymous?"). A legal assistant must handle both, so we run both searches and merge them. The BM25 side uses a simple tokenizer: lowercase, split on anything that is not a letter or digit, keep umlauts. No stemming yet.

**How the merge works.** LangChain's `EnsembleRetriever` uses *Reciprocal Rank Fusion*. It ignores the raw scores (BM25 and cosine scores are on different scales) and only looks at positions. A passage at rank r in a list earns `weight / (r + 60)` points; points from both lists are added. A passage that appears high in both lists wins; one that only one list found can still make it, but with fewer points. Our weights are 0.4 for BM25 and 0.6 for vectors, and each side fetches 12 candidates so the merge has material to work with.

**Why we de-duplicate.** The index holds every provision in German *and* English, and long provisions are split into parts. Without cleanup the top 6 could be three copies of Art. 5. So results that share the same identity (regulation, type, number, paragraph) collapse into one. Which copy survives? The one in the question's language, so a German question gets German text to quote.

**What the notebook showed (5 questions).** Hybrid found the expected provision every time, but it was not magic. BM25 alone missed Recital 26 for the German anonymisation question; vector alone found it. For the "Artikel 5" question BM25 only reached rank 6 and the hybrid put Art. 5 at rank 4, behind the vector-only rank 3, because common words like "KI" and "Verordnung" make the number a weak signal. Hybrid helped most when both lists agreed (lawfulness of processing: rank 3 vs 5 and 6). Honest takeaway: hybrid is a safety net that is rarely much worse than the better single method. Five questions are only a feeling; stage 6 measures it properly.

**Check yourself:** (1) Why does RRF use ranks instead of scores? (2) What would happen to a German question if we skipped de-duplication? (3) Which question type would you expect BM25 to beat embeddings on, and why did the Art. 5 example not show that clearly?

## §5 RAG chain

**The idea.** Retrieval gives us the right passages; the chain makes an LLM turn them into a readable answer *without* letting it improvise. This is "grounding": the model is a writer, not a source of facts. The facts come from the passages we hand it.

**What the prompt does.** It is deliberately strict. We number the passages (`[1] AI Act Art. 6(2) (en): ...`) and tell the model: answer only from these, write in German or English, put a marker like `[2]` after every claim, and if the passages don't answer the question, reply with exactly `NOT_COVERED`. The numbers matter: they give the model a short, unambiguous way to point at its source, and give us something we can check by machine. The language is detected from the question with `langdetect`. A French question is answered in English: we only promise German and English.

**From `[n]` to citations.** After the model answers, `parse_citations` scans the text for markers. A marker is valid only if its number is between 1 and the number of passages we sent. Valid ones are mapped back to the passage's metadata and become citations (label like "GDPR Recital 26" plus the EUR-Lex link), listed once each in order of first use. Invalid ones, like `[9]` when we sent 6 passages, are removed. Grouped markers like `[1, 3]` are rewritten as `[1][3]`.

**Why refuse.** An LLM that cannot find an answer will often write a confident one anyway. We catch that in two ways: the model itself says `NOT_COVERED`, or its answer contains no valid citation at all. In both cases we throw the text away and return a fixed "not covered" message with no citations. An uncited answer is never shown as if it were trustworthy. The cost: sometimes a correct but uncited answer is refused. For a legal assistant that is the safer error.

**Temperature 0.** Temperature controls how adventurous the model is when picking the next word. At 0 it always picks the most likely one, so the same question gives (nearly) the same answer, which also makes evaluation in stage 6 fairer.

**Tokens and latency.** Models read and write *tokens* (word pieces) and providers bill and rate-limit by them. We record input and output tokens from the model's reply metadata (0 if absent) and the wall-clock time of retrieval plus the model call. Both go into the `Answer` object, so the API can show them and we can watch cost.

**What `prompt | llm` means.** This is LCEL (LangChain Expression Language). The `|` pipes the output of one step into the next, like a shell pipe: the prompt template fills in the placeholders and produces messages, and those messages flow into the model. Any step with `invoke` can join the pipe, which is why tests can swap in a fake model.

**Check yourself:** (1) Why number the passages instead of asking the model to name the article? (2) What happens to an answer whose only marker is `[9]`? (3) Why is refusing a correct-but-uncited answer acceptable here?

## §6 Evaluation

**Notebook:** `notebooks/06_evaluation.ipynb` | **Code:** `eval/run_eval.py`, `eval/metrics.py` | **Data:** `eval/gold.jsonl`, results in `eval/results.md`

**Why evaluate.** Trying a few questions by hand gives a feeling. An evaluation asks the same fixed questions every time and turns the results into numbers, so we can compare designs (stage 2) and see when a change makes things worse.

**The gold set.** 40 questions, 20 German and 20 English: 15 AI Act, 15 GDPR, 5 that need both laws, and 5 out of scope (tax, minimum wage, the Digital Markets Act ...) that the assistant must refuse. For each question we wrote down the provision(s) that answer it, plus a `note` saying why, and checked every one against the actual chunk text, not from memory. That mattered: in the consolidated AI Act the bias-detection rule moved from Art. 10(5) to a new Art. 4a. **The gold set was reviewed by the author on 2026-11-23, with no changes.**

**The metrics.**
- *hit@k (article / paragraph):* is an expected provision among the top 6 results? Retrieval is the ceiling: if the right text is not retrieved, the LLM cannot answer from it.
- *citation correctness:* does the answer cite an expected article?
- *refusal accuracy:* does it refuse exactly the out-of-scope questions?
- *faithfulness and relevance (1-5):* a second LLM call, the "judge", grades whether every claim is supported by the passages and whether the question was answered.
- *p50 / p95 latency:* the time half / 95% of requests stay under. Unlike the average, p95 shows the slow cases.

**Results so far (retrieval only, no LLM).** Structured: hit@k article 82.9% (n = 35), paragraph 64.3% (n = 28), retrieval 26 ms median / 30 ms p95 (after the model is loaded). Answer metrics are pending: they need a `GOOGLE_API_KEY`. The run is built for the free tier: it waits 6 s between LLM calls, caches every answer and verdict, and resumes after a quota error without asking anything twice.

**What the misses taught us.** In 5 of the 6 misses the top results were *recitals*. Recitals explain the law in plain language that sounds like a question, so they beat the formal article. Sometimes the recital would even give a correct answer (GDPR Recital 51 lists the same sensitive categories as Art. 9(1)), just with a weaker citation. We did not tune anything to fix this: changing settings until the same 35 questions score better would only "overfit the test". Possible next steps, to decide together: a small boost for articles, or accepting a recital where it really answers the question.

**What the numbers do not prove.** 35 questions is small: the 95% confidence interval for 82.9% is roughly 67-92%. We wrote the questions ourselves, so they may be easier than real users' questions. And a retrieval hit is not a good answer; that is what the pending metrics are for.

**Check yourself:** (1) Why is hit@k a ceiling for answer quality? (2) Why do out-of-scope questions not count in hit@k but do count in refusal accuracy? (3) Why is it a bad idea to tune top_k on the gold set and report the tuned score?

## §7 API + UI

**Code:** `src/assistant/api.py` | **Tests:** `tests/test_api.py`

### API

**What FastAPI does here.** FastAPI turns a Python function into a web endpoint. `POST /ask` takes `{"question": "..."}` and returns the `Answer` (text, citations, `refused`, latency, tokens); `GET /health` says the service is up, how many chunks are indexed and which version of the laws the text is from ("AI Act 2026-07-27; GDPR 2016-05-04"). `create_app()` builds the app; in production it loads the embedding model, index, retriever and LLM once at startup, in tests we hand it a fake `answer_fn` so no model or key is needed. If no API key is set, the app still starts: `/health` works and `/ask` answers 503.

**Why three guards.** A public demo on a free-tier key has a fixed quota, and anyone on the internet can call the endpoint. Without limits one script could use up the day's quota in a minute and the demo would be dead for everyone. So: (1) a *rate limit* of 10 questions per IP per hour, as a sliding window (each request is remembered for 3600 seconds); (2) a *daily cap* of 200 model calls in total, which resets at midnight UTC; (3) an *answer cache* (the last 256 questions, matched ignoring case and surrounding spaces), so the same question never costs a second model call. A cached answer still counts toward the IP limit but not toward the daily cap, because only the model call costs quota. Everything is kept in memory, which is fine for one process and is lost on restart. A failed provider call still counts against the daily cap: this protects the free-tier quota during provider trouble, at the price that an outage can use up the cap.

**Which IP to trust.** The client's address arrives in the `X-Forwarded-For` header, but a caller can put anything at the *start* of that header, so the first entry is spoofable: sending a fresh fake one per request would dodge the per-IP limit. Each trusted reverse proxy *appends* the address it actually saw, so we read the entry `TRUSTED_PROXY_HOPS` places from the right (default 1 = the last entry, as written by the Hugging Face Spaces proxy; 0 = ignore the header and use the socket address). The per-IP table is capped at 10,000 entries, dropping the least recently seen IP, so it cannot grow without bound. Even so, the daily cap is the real backstop for spend: whatever happens to the per-IP limit, the model is never called more than `DAILY_CAP` times a day.

**What the status codes mean.**
- **422**: the question is empty, only spaces, or longer than 1000 characters. The message says which.
- **429**: too many questions from your IP ("Too many questions, please try again later.") or the demo's daily cap is reached ("... daily limit, please try again tomorrow.").
- **503**: the model call failed, usually because the provider quota is used up. The caller sees one friendly message, never a traceback or the key; the log records only the error type. Failed calls are not cached.

**Check yourself:** (1) Why does a cache hit not count toward the daily cap, but still toward the IP limit? (2) What could a caller do to dodge the per-IP limit, and what still protects the quota? (3) Why must the app start without an API key?

### UI

(Added in the next step: the Streamlit UI.)

## §8 Docker, CI, deployment
