"""Tests for agreement.py: Cohen's kappa, label validation, compare and merge."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agreement import cohen_kappa, compare, load_labels, main, make_blind, merge  # noqa: E402

GOLD = ROOT / "data" / "gold" / "sample_gold.csv"


def sheet(rows):
    return pd.DataFrame(rows, columns=["sentence", "topic", "sentiment"])


def test_kappa_perfect_and_chance():
    assert cohen_kappa(["a", "b", "a", "b"], ["a", "b", "a", "b"]) == 1.0
    # Agreement exactly at chance level gives 0.
    assert cohen_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)


def test_kappa_known_value():
    # Textbook 2x2 example: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> kappa = 0.4
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_same_single_label():
    assert cohen_kappa(["a", "a"], ["a", "a"]) == 1.0


def test_load_labels_rejects_blank_and_unknown(tmp_path):
    p = tmp_path / "bad.csv"
    sheet([["s1", "food", "positive"], ["s2", "", "negative"], ["s3", "drinks", "meh"]]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="3 invalid"):
        load_labels(p)


def test_load_labels_normalizes_case_and_spaces(tmp_path):
    p = tmp_path / "ok.csv"
    sheet([["s1", " Food ", "POSITIVE"]]).to_csv(p, index=False)
    df = load_labels(p)
    assert df.loc[0, "topic"] == "food" and df.loc[0, "sentiment"] == "positive"


def test_compare_and_merge():
    a = sheet([["s1", "food", "positive"], ["s2", "service", "negative"], ["s3", "price", "negative"]])
    b = sheet([["s1", "food", "positive"], ["s2", "wait_time", "negative"], ["s3", "price", "neutral"]])
    table, dis = compare(a, b)
    assert table.set_index("label").loc["topic", "agreement"] == pytest.approx(0.667, abs=1e-3)
    assert dis["sentence"].tolist() == ["s2", "s3"]

    with pytest.raises(ValueError, match="2 disagreement"):
        merge(a, b, dis)  # not resolved yet

    dis["final_topic"] = ["wait_time", "price"]
    dis["final_sentiment"] = ["negative", "negative"]
    gold = merge(a, b, dis)
    assert gold["topic"].tolist() == ["food", "wait_time", "price"]
    assert gold["sentiment"].tolist() == ["positive", "negative", "negative"]


def test_compare_requires_same_sentences():
    a = sheet([["s1", "food", "positive"]])
    b = sheet([["other", "food", "positive"]])
    with pytest.raises(ValueError, match="sentences differ"):
        compare(a, b)


def test_blind_sheet_matches_gold():
    blind = make_blind(GOLD)
    gold = pd.read_csv(GOLD)
    assert blind["sentence"].tolist() == gold["sentence"].tolist()
    assert (blind["topic"] == "").all() and (blind["sentiment"] == "").all()


def test_cli_compare_gold_with_itself(tmp_path, capsys):
    out = tmp_path / "dis.csv"
    assert main(["compare", str(GOLD), str(GOLD), "--out", str(out)]) == 0
    assert "almost perfect" in capsys.readouterr().out
    assert len(pd.read_csv(out)) == 0
