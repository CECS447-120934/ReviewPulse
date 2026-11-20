import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data import load_csv, load_reviews, load_yelp  # noqa: E402
from preprocess import clean_text, scrub, split_sentences, to_sentences  # noqa: E402


# --- splitting ---------------------------------------------------------------
def test_split_protects_abbreviations_and_decimals():
    s = "Dr. Smith said the soup was $12.50 there. We loved it, e.g. the bread was warm! We will return."
    assert split_sentences(s) == [
        "Dr. Smith said the soup was $12.50 there.",
        "We loved it, e.g. the bread was warm!",
        "We will return.",
    ]


def test_split_keeps_ellipses_and_drops_short_fragments():
    s = "We waited 30 minutes for drinks... and another 30 for food. Wow! Never again would we go."
    assert split_sentences(s) == ["We waited 30 minutes for drinks... and another 30 for food.",
                                  "Never again would we go."]


def test_split_on_newlines():
    assert split_sentences("Great pasta tonight\nThe service was slow") == ["Great pasta tonight",
                                                                          "The service was slow"]


# --- cleaning / scrubbing -------------------------------------------------------
def test_clean_text_normalizes():
    assert clean_text("It’s  &amp; great \n\n here") == "It's & great\nhere"


@pytest.mark.parametrize("text, expected", [
    ("Our server Jessica was great.", "Our server [NAME] was great."),
    ("The manager, Tom, came by.", "The manager, [NAME], came by."),
    ("Jessica was our waitress tonight.", "[NAME] was our waitress tonight."),
    ("Tom (the manager) helped.", "[NAME] (the manager) helped."),
    ("Ask for Danielle at the bar.", "Ask for [NAME] at the bar."),
    ("Shoutout to Marco for the drinks.", "Shoutout to [NAME] for the drinks."),
    ("Dr. Patel sent us here.", "[NAME] sent us here."),
    ("Call (562) 555-0143 or 562.555.0143.", "Call [PHONE] or [PHONE]."),
    ("Email feedback@example.com now.", "Email [EMAIL] now."),
    ("See www.example.com/menu today.", "See [URL] today."),
    ("Review id abcDEF123ghiJKL456mnoP here.", "Review id [ID] here."),
])
def test_scrub(text, expected):
    assert scrub(text) == expected


@pytest.mark.parametrize("text", [
    "Great food, our server was attentive.",   # lowercase word before role is not a name
    "Our server was very attentive.",          # common word after role
    "The server The food came fast.",          # capitalized stopword after role
    "We waited 45 minutes for a table.",
])
def test_scrub_leaves_non_names(text):
    assert scrub(text) == text


# --- loading ------------------------------------------------------------------
def test_load_csv_aliases_and_ids(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("Created At,Rating,Review Text\n2026-10-05,4,Good pasta.\n2026-10-06,,  \nbad-date,3,Hm ok.\n")
    df = load_csv(p, business_id="demo")
    assert list(df.columns) == ["review_id", "business_id", "stars", "date", "text"]
    assert len(df) == 1  # blank text and unparseable date dropped
    assert df.loc[0, "business_id"] == "demo" and df.loc[0, "stars"] == 4.0
    assert df.loc[0, "review_id"] == load_csv(p, business_id="demo").loc[0, "review_id"]  # stable


def test_load_csv_requires_text(tmp_path):
    p = tmp_path / "r.csv"
    p.write_text("date,stars\n2026-10-05,4\n")
    with pytest.raises(ValueError, match="text column"):
        load_csv(p)


def _write_yelp(tmp_path):
    rows = [
        {"review_id": "r1", "user_id": "u", "business_id": "B1", "stars": 5, "text": "Great food here.", "date": "2026-10-01 12:00:00"},
        {"review_id": "r2", "user_id": "u", "business_id": "B2", "stars": 1, "text": "Bad service here.", "date": "2026-10-02 12:00:00"},
        {"review_id": "r3", "user_id": "u", "business_id": "B1", "stars": 3, "text": "Okay food here.", "date": "2026-10-03 12:00:00"},
    ]
    (tmp_path / "yelp_academic_dataset_review.json").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


@pytest.mark.parametrize("load_all", [False, True])
def test_load_yelp_filters_and_limits(tmp_path, load_all):
    _write_yelp(tmp_path)
    df = load_yelp(tmp_path, business_ids="B1", load_all=load_all)
    assert df["review_id"].tolist() == ["r1", "r3"]
    assert load_yelp(tmp_path, limit=2, load_all=load_all)["review_id"].tolist() == ["r1", "r2"]


# --- end to end ---------------------------------------------------------------
def test_to_sentences_on_sample():
    reviews = load_reviews(ROOT / "data" / "sample_reviews.csv")
    sents = to_sentences(reviews)
    assert list(sents.columns) == ["review_id", "business_id", "stars", "date", "week", "sent_idx", "sentence"]
    assert len(sents) > len(reviews)
    assert (sents["week"].dt.day_of_week == 0).all()          # weeks start on Monday
    assert (sents["week"] <= sents["date"]).all()
    assert sents["week"].nunique() == 6
    joined = " ".join(sents["sentence"])
    for leaked in ("Jessica", "Marco", "Danielle", "Kevin", "555-0143", "feedback@example.com"):
        assert leaked not in joined
