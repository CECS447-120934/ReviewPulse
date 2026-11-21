"""Tests for labeling, caching, ranking, the brief and replies. No network: the Claude client is faked."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import llm  # noqa: E402
from brief import check_grounding, template_brief, write_brief  # noqa: E402
from data import load_reviews  # noqa: E402
from evaluate import macro_f1  # noqa: E402
from labelers import KeywordLabeler, LLMLabeler, parse_llm_json  # noqa: E402
from pipeline import analyze  # noqa: E402
from replies import draft_reply, promise_check  # noqa: E402
from store import LabelCache  # noqa: E402

SAMPLE = ROOT / "data" / "sample_reviews.csv"


class FakeClient:
    """Mimics client.beta.messages.create. Replies with a thinking block first, like Opus 5.5 does."""

    def __init__(self, reply_fn, stop_reason="end_turn"):
        self.reply_fn, self.stop_reason, self.calls = reply_fn, stop_reason, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        content = [SimpleNamespace(type="thinking", thinking=""),
                   SimpleNamespace(type="text", text=self.reply_fn(kw))]
        return SimpleNamespace(content=content, stop_reason=self.stop_reason,
                               usage=SimpleNamespace(input_tokens=1000, output_tokens=200))


def _label_all_food(kw):
    items = json.loads(kw["messages"][0]["content"].removeprefix("Input: ").removesuffix("\nOutput:"))
    return json.dumps([{"i": it["i"], "topic": "food", "sentiment": "negative", "detail": "x"} for it in items])


# --- labelers -----------------------------------------------------------------
def test_parse_llm_json_rejects_bad_labels_and_tolerates_text():
    text = 'Here you go:\n[{"i":1,"topic":"food","sentiment":"positive","detail":"good"},' \
           '{"i":2,"topic":"parking","sentiment":"angry"}]'
    out = parse_llm_json(text, 3)
    assert out[0] == {"topic": "food", "sentiment": "positive", "detail": "good"}
    assert out[1]["topic"] == "none" and out[1]["sentiment"] == "neutral"
    assert out[2]["topic"] == "none"
    assert parse_llm_json("not json", 2) == [parse_llm_json("", 1)[0]] * 2


@pytest.mark.parametrize("sentence, topic, sentiment", [
    ("We waited 40 minutes for a table.", "wait_time", "neutral"),
    ("The bathroom was dirty.", "cleanliness", "negative"),
    ("The steak was not bland at all.", "food", "positive"),
    ("We went here for my birthday.", "none", "neutral"),
])
def test_keyword_labeler(sentence, topic, sentiment):
    lab = KeywordLabeler().label([sentence])[0]
    assert (lab["topic"], lab["sentiment"]) == (topic, sentiment)


def test_llm_labeler_batches_skips_thinking_blocks_and_caches():
    cache = LabelCache(":memory:")
    client = FakeClient(_label_all_food)
    lab = LLMLabeler(client=client, cache=cache, batch_size=2, model="test-model")
    sents = ["a b c", "d e f", "g h i"]
    assert [l["topic"] for l in lab.label(sents)] == ["food"] * 3
    assert len(client.calls) == 2 and len(cache) == 3
    lab.label(sents)  # all cached now
    assert len(client.calls) == 2


def test_llm_labeler_falls_back_on_refusal_without_caching():
    cache = LabelCache(":memory:")
    lab = LLMLabeler(client=FakeClient(_label_all_food, stop_reason="refusal"), cache=cache, model="m")
    out = lab.label(["The bathroom was dirty."])
    assert out[0]["topic"] == "cleanliness"  # keyword fallback
    assert lab.failed_batches == 1 and len(cache) == 0
    assert "LLMRefusal" in lab.last_error


def test_fallback_is_reported_not_silent():
    from evaluate import evaluate

    lab = LLMLabeler(client=FakeClient(_label_all_food, stop_reason="refusal"), model="m")
    result = analyze(load_reviews(SAMPLE), lab)
    assert "failed and used keyword labels" in result.fallback_note
    gold = pd.read_csv(ROOT / "data" / "gold" / "sample_gold.csv")
    row, _ = evaluate(gold, LLMLabeler(client=FakeClient(_label_all_food, stop_reason="refusal"), model="m"))
    assert "INVALID" in row["labeler"] and row["failed_batches"] > 0
    assert analyze(load_reviews(SAMPLE), KeywordLabeler()).fallback_note == ""


# --- ranking ------------------------------------------------------------------
def test_sample_ranks_wait_time_first_with_evidence():
    result = analyze(load_reviews(SAMPLE), KeywordLabeler())
    assert result.problems[0]["topic"] == "wait_time"
    assert len(result.problems) <= 3
    labeled_sents = set(result.labeled["sentence"])
    for p in result.problems:
        assert p["evidence"] and all(e["sentence"] in labeled_sents for e in p["evidence"])
        assert p["score"] > 0


@pytest.mark.parametrize("csv", ["date,text\n2026-11-01,Wow!\n", "date,text\n2026-11-01,\n"])
def test_no_usable_sentences_gives_clear_error(tmp_path, csv):
    from data import load_csv

    p = tmp_path / "r.csv"
    p.write_text(csv)
    with pytest.raises(ValueError, match="no usable review sentences"):
        analyze(load_csv(p), KeywordLabeler())


def test_ranking_rewards_rising_complaints():
    # Same number of complaints now, but topic A had none before and B had the same amount before.
    rows = []
    def add(week, topic, sentiment, n):
        for k in range(n):
            d = pd.Timestamp("2026-09-07") + pd.Timedelta(weeks=week)
            rows.append({"review_id": f"{topic}{week}{k}", "business_id": "b", "stars": 2.0, "date": d,
                         "week": d, "sent_idx": 0, "sentence": f"{topic} complaint {week} {k}",
                         "topic": topic, "sentiment": sentiment, "detail": ""})
    for w in range(4):
        add(w, "price", "negative", 1)       # steady before
    for w in range(4, 8):
        add(w, "price", "negative", 1)       # steady now
        add(w, "cleanliness", "negative", 1)  # new now
    from ranking import rank_problems
    top = rank_problems(pd.DataFrame(rows))
    assert [p["topic"] for p in top][:2] == ["cleanliness", "price"]


# --- brief --------------------------------------------------------------------
def test_grounding_flags_invented_quotes():
    problems = [{"evidence": [{"sentence": "Our pizza took 35 minutes to arrive."}]}]
    assert check_grounding('Guests said "our pizza took 35 minutes to arrive".', problems) == []
    assert check_grounding('Guests said "the chef yelled at everyone tonight".', problems) == [
        "the chef yelled at everyone tonight"]


def test_brief_llm_and_template_paths():
    result = analyze(load_reviews(SAMPLE), KeywordLabeler())
    quote = result.problems[0]["evidence"][0]["sentence"]
    good = write_brief(result.summary, result.problems, client=FakeClient(lambda kw: f'1. **Wait** "{quote}"'))
    assert good["source"].startswith("llm:") and good["unsupported_quotes"] == []
    bad = write_brief(result.summary, result.problems, client=FakeClient(lambda kw: '"A made-up complaint here"'))
    assert bad["unsupported_quotes"] == ["A made-up complaint here"]
    offline = write_brief(result.summary, result.problems)
    assert offline["source"] == "template" and offline["unsupported_quotes"] == []
    assert template_brief(result.summary, []).endswith("No recurring problems found in this window.")


# --- replies ------------------------------------------------------------------
def test_replies():
    assert promise_check("We'd love to offer a full refund and a free dessert.") == ["free", "refund"]
    r = draft_reply("Slow service.", 2.0, ["wait_time"], client=FakeClient(lambda kw: "Sorry about the wait."))
    assert r["text"] == "Sorry about the wait." and r["source"].startswith("llm:")
    r = draft_reply("Slow service.", 2.0, ["wait_time"], client=FakeClient(lambda kw: "x", stop_reason="refusal"))
    assert r["source"] == "template" and "wait time" in r["text"]


# --- evaluation / cost --------------------------------------------------------
def test_macro_f1_and_cost():
    assert macro_f1(["a", "b"], ["a", "b"], ["a", "b"]) == 1.0
    assert macro_f1(["a", "a"], ["b", "b"], ["a", "b"]) == 0.0
    assert llm.usage_cost({"calls": 1, "input_tokens": 1_000_000, "output_tokens": 1_000_000},
                          model="claude-opus-5-5") == pytest.approx(24.0)
