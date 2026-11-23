import json

import pytest
from langchain_core.documents import Document

from assistant.models import Answer, Chunk, Citation
from eval.run_eval import (
    PENDING,
    Throttle,
    missing_expectations,
    parse_judge_output,
    render_appendix,
    render_results,
    run,
    summarize,
    validate_gold,
)

GOLD = [
    {
        "id": "q01",
        "question": "When is processing lawful?",
        "language": "en",
        "expected": [{"regulation": "GDPR", "type": "article", "number": "6", "paragraph": "1"}],
        "out_of_scope": False,
        "note": "Art. 6(1) lists the legal bases.",
    },
    {
        "id": "q02",
        "question": "Is Recital 26 about anonymous data?",
        "language": "en",
        "expected": [{"regulation": "GDPR", "type": "recital", "number": "26", "paragraph": None}],
        "out_of_scope": False,
        "note": "Recital 26.",
    },
    {
        "id": "q03",
        "question": "What is the German minimum wage?",
        "language": "en",
        "expected": [],
        "out_of_scope": True,
        "note": "Not in either regulation.",
    },
]


def doc(number, paragraph="", typ="article", reg="GDPR", chunk_id=None):
    return Document(
        page_content=f"Article {number}\ntext",
        metadata={
            "regulation": reg,
            "type": typ,
            "number": number,
            "paragraph": paragraph,
            "language": "en",
            "title": "",
            "source_url": "https://eur-lex.europa.eu/x",
            "chunk_id": chunk_id or f"{reg}-{number}-{paragraph}",
        },
    )


def fake_retrieve(question, language):
    # Art. 6(2) at rank 1 and Art. 6(1) at rank 2 for q01; nothing useful for the others.
    if "lawful" in question:
        return [doc("6", "2"), doc("6", "1")]
    return [doc("99", "1")]


def make_answer(question, refused=False, cite=("GDPR", "article", "6", "1")):
    citations = (
        []
        if refused
        else [
            Citation(
                regulation=cite[0],
                type=cite[1],
                number=cite[2],
                paragraph=cite[3],
                source_url="https://eur-lex.europa.eu/x",
                language="en",
            )
        ]
    )
    return Answer(
        answer="It is lawful if ... [1]",
        language="en",
        citations=citations,
        refused=refused,
        latency_ms=1200,
        input_tokens=900,
        output_tokens=100,
    )


class Recorder:
    """Fake answer function and judge that count their calls."""

    def __init__(self, judge_result=None, fail_answer_after=None):
        self.answers = 0
        self.judges = 0
        self.judge_result = judge_result or {"faithfulness": 5, "relevance": 4, "reason": "ok"}
        self.fail_answer_after = fail_answer_after

    def answer(self, question):
        if self.fail_answer_after is not None and self.answers >= self.fail_answer_after:
            raise RuntimeError("429 quota exceeded")
        self.answers += 1
        return make_answer(question, refused="minimum wage" in question)

    def judge(self, question, answer_text, passages):
        self.judges += 1
        return self.judge_result


class NoSleep(Throttle):
    def __init__(self, interval=6.0):
        self.slept = []
        self.now = 0.0
        super().__init__(interval, sleep=self.slept.append, clock=lambda: self.now)


def test_throttle_waits_between_calls_but_not_before_the_first():
    t = NoSleep(6.0)
    t.wait()
    assert t.slept == []
    t.now = 2.0
    t.wait()
    assert t.slept == [4.0]


def test_retrieval_only_makes_no_llm_calls_and_scores_hits(tmp_path):
    rec = Recorder()
    records = run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=rec.answer,
        judge_fn=rec.judge,
        retrieval_only=True,
        throttle=NoSleep(),
    )
    assert rec.answers == 0 and rec.judges == 0
    assert not list(tmp_path.iterdir())  # nothing cached in retrieval-only mode
    by_id = {r["id"]: r for r in records}
    assert by_id["q01"]["hit_article"] is True
    assert by_id["q01"]["hit_paragraph"] is True
    assert by_id["q01"]["rank_article"] == 1 and by_id["q01"]["rank_paragraph"] == 2
    assert by_id["q02"]["hit_article"] is False and by_id["q02"]["hit_paragraph"] is None
    assert by_id["q03"]["hit_article"] is None  # out of scope: not scored
    assert all(r["retrieval_ms"] >= 0 for r in records)


def test_answers_and_verdicts_are_cached_and_reused(tmp_path):
    rec = Recorder()
    run(GOLD, fake_retrieve, tmp_path, answer_fn=rec.answer, judge_fn=rec.judge, throttle=NoSleep())
    assert rec.answers == 3
    assert rec.judges == 2  # the refused out-of-scope answer is not judged
    cached = json.loads((tmp_path / "q01.json").read_text())
    assert cached["question"] == GOLD[0]["question"]
    assert cached["judge"]["faithfulness"] == 5

    again = Recorder()
    records = run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=again.answer,
        judge_fn=again.judge,
        throttle=NoSleep(),
    )
    assert again.answers == 0 and again.judges == 0  # resumed entirely from the cache
    assert records[0]["answer"]["citations"][0]["number"] == "6"


def test_changed_question_invalidates_its_cache_entry(tmp_path):
    run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=Recorder().answer,
        judge_fn=Recorder().judge,
        throttle=NoSleep(),
    )
    edited = [dict(GOLD[0], question="When is processing lawful under the GDPR?")] + GOLD[1:]
    rec = Recorder()
    run(
        edited,
        fake_retrieve,
        tmp_path,
        answer_fn=rec.answer,
        judge_fn=rec.judge,
        throttle=NoSleep(),
    )
    assert rec.answers == 1 and rec.judges == 1


def test_skip_judge_then_judge_later_reuses_answers(tmp_path):
    rec = Recorder()
    run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=rec.answer,
        judge_fn=rec.judge,
        skip_judge=True,
        throttle=NoSleep(),
    )
    assert rec.answers == 3 and rec.judges == 0
    later = Recorder()
    run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=later.answer,
        judge_fn=later.judge,
        throttle=NoSleep(),
    )
    assert later.answers == 0 and later.judges == 2


def test_unparseable_verdict_is_not_cached_so_a_rerun_retries(tmp_path):
    bad = Recorder(judge_result=None)
    bad.judge_result = None
    bad.judge = lambda q, a, p: None
    records = run(
        GOLD, fake_retrieve, tmp_path, answer_fn=bad.answer, judge_fn=bad.judge, throttle=NoSleep()
    )
    assert records[0]["judge"] is None and records[0]["judge_error"]
    rec = Recorder()
    run(GOLD, fake_retrieve, tmp_path, answer_fn=rec.answer, judge_fn=rec.judge, throttle=NoSleep())
    assert rec.answers == 0 and rec.judges == 2


def test_provider_error_stops_llm_calls_but_keeps_retrieval_and_cache(tmp_path):
    rec = Recorder(fail_answer_after=1)
    records = run(
        GOLD, fake_retrieve, tmp_path, answer_fn=rec.answer, judge_fn=rec.judge, throttle=NoSleep()
    )
    assert rec.answers == 1
    assert records[0]["answer"] is not None
    assert records[1]["answer"] is None and "429" in records[1]["llm_error"]
    assert records[2]["answer"] is None
    assert all("hit_article" in r for r in records)
    assert (tmp_path / "q01.json").exists() and not (tmp_path / "q02.json").exists()


def test_throttle_is_applied_only_to_real_llm_calls(tmp_path):
    throttle = NoSleep()
    run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=Recorder().answer,
        judge_fn=Recorder().judge,
        throttle=throttle,
    )
    assert len(throttle.slept) == 3 + 2 - 1  # every call but the first waited
    resumed = NoSleep()
    run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=Recorder().answer,
        judge_fn=Recorder().judge,
        throttle=resumed,
    )
    assert resumed.slept == []


def test_summarize_and_render(tmp_path):
    records = run(
        GOLD,
        fake_retrieve,
        tmp_path,
        answer_fn=Recorder().answer,
        judge_fn=Recorder().judge,
        throttle=NoSleep(),
    )
    s = summarize(records)
    assert s["n_in_scope"] == 2 and s["n_out_of_scope"] == 1
    assert s["hit_article"] == {"value": 0.5, "n": 2}
    assert s["hit_paragraph"] == {"value": 1.0, "n": 1}
    assert s["refusal_accuracy"] == {"value": 1.0, "n": 3}
    assert s["citation_correct"] == {"value": 0.5, "n": 2}  # q02 cites 6(1), expected Rec. 26
    assert s["faithfulness"] == {"value": 5.0, "n": 2}
    assert s["avg_tokens"] == {"value": 1000.0, "n": 3}
    md = render_results({"structured": {"summary": s, "meta": {"date": "2026-11-23"}}})
    assert "hit@k (article)" in md and "50.0% (n=2)" in md


def test_render_marks_answer_metrics_pending_after_retrieval_only(tmp_path):
    records = run(GOLD, fake_retrieve, tmp_path, retrieval_only=True, throttle=NoSleep())
    s = summarize(records)
    assert s["faithfulness"] is None
    md = render_results({"structured": {"summary": s, "meta": {}}})
    assert PENDING in md
    appendix_md, appendix_csv = render_appendix({"structured": {"records": records}}, GOLD)
    assert "q01" in appendix_md and "GDPR Art. 6(1)" in appendix_md
    assert appendix_csv.splitlines()[0].startswith("id,")


def test_naive_coverage_counts_text_that_was_retrieved_under_another_label(tmp_path):
    # The retrieved naive piece starts in Art. 5 but its text runs into Art. 6(1).
    coverage = {"GDPR-6-2": [("GDPR", "article", "5", "3"), ("GDPR", "article", "6", "1")]}

    def retrieve(question, language):
        return [doc("5", "3", chunk_id="GDPR-6-2")]

    records = run(
        GOLD[:1], retrieve, tmp_path, retrieval_only=True, coverage=coverage, throttle=NoSleep()
    )
    assert records[0]["hit_article"] is False
    assert records[0]["hit_article_text"] is True


@pytest.mark.parametrize(
    "raw",
    [
        '{"faithfulness": 4, "relevance": 5, "reason": "fine"}',
        '```json\n{"faithfulness": 4, "relevance": 5, "reason": "fine"}\n```',
        'Here is my verdict: {"faithfulness": "4", "relevance": 5.0, "reason": "fine"} Thanks.',
    ],
)
def test_parse_judge_output_accepts_common_shapes(raw):
    assert parse_judge_output(raw) == {"faithfulness": 4, "relevance": 5, "reason": "fine"}


@pytest.mark.parametrize(
    "raw", ["no json here", '{"faithfulness": 7, "relevance": 5}', '{"relevance": 3}', "{broken"]
)
def test_parse_judge_output_rejects_invalid(raw):
    assert parse_judge_output(raw) is None


def test_validate_gold_rejects_bad_items():
    validate_gold(GOLD)
    with pytest.raises(ValueError, match="duplicate"):
        validate_gold([GOLD[0], GOLD[0]])
    with pytest.raises(ValueError, match="out_of_scope"):
        validate_gold([dict(GOLD[2], expected=GOLD[0]["expected"])])
    with pytest.raises(ValueError, match="expected"):
        validate_gold([dict(GOLD[0], expected=[])])
    with pytest.raises(ValueError, match="language"):
        validate_gold([dict(GOLD[0], language="fr")])


def test_missing_expectations_finds_provisions_not_in_the_corpus():
    def chunk(number, paragraph, lang):
        return Chunk(
            id=f"x-{number}-{paragraph}-{lang}",
            text="t",
            regulation="GDPR",
            language=lang,
            type="article",
            number=number,
            paragraph=paragraph,
            source_url="u",
            celex="c",
            version_date="d",
        )

    chunks = [chunk("6", "1", "en"), chunk("6", "1", "de")]
    article_level = dict(GOLD[0], id="q04", expected=[dict(GOLD[0]["expected"][0], paragraph=None)])
    missing = missing_expectations(GOLD + [article_level], chunks)
    assert missing == [("q02", "GDPR recital 26")]  # Art. 6 without paragraph matches Art. 6(1)
