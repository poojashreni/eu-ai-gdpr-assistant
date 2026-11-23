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

## §5 RAG chain

## §6 Evaluation

## §7 API + UI

## §8 Docker, CI, deployment
