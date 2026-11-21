"""Rank problems from labeled sentences (no AI here: plain, explainable arithmetic).

For each topic, over the most recent `window` weeks:
    mentions   = sentences labeled with the topic
    negativity = mean sentiment weight (negative 1.0, neutral 0.25, positive 0.0)
    trend      = (negative mentions this window - previous window) / max(previous, 1), clipped to [-0.5, 2]
    score      = mentions x negativity x (1 + trend)
The top three topics with a positive score become the brief's problems. Each problem carries quoted
evidence (the actual sentences and their review IDs), so nothing in the brief is unsupported.
"""
import pandas as pd

from schema import PROBLEM_TOPICS

NEG_WEIGHT = {"negative": 1.0, "neutral": 0.25, "positive": 0.0}
TREND_CLIP = (-0.5, 2.0)


def attach_labels(sentences: pd.DataFrame, labels: list[dict]) -> pd.DataFrame:
    """Join labeler output (one dict per sentence, same order) onto the sentence table."""
    lab = pd.DataFrame(labels, index=sentences.index)[["topic", "sentiment", "detail"]]
    return pd.concat([sentences, lab], axis=1)


def weekly_counts(labeled: pd.DataFrame, sentiment: str = "negative") -> pd.DataFrame:
    """Week x topic table of sentence counts with the given sentiment (rows: every week in the data)."""
    weeks = pd.Index(sorted(labeled["week"].unique()), name="week")
    sub = labeled[(labeled["sentiment"] == sentiment) & (labeled["topic"].isin(PROBLEM_TOPICS))]
    table = sub.pivot_table(index="week", columns="topic", values="sentence", aggfunc="count", fill_value=0)
    return table.reindex(index=weeks, columns=PROBLEM_TOPICS, fill_value=0).astype(int)


def rank_problems(labeled: pd.DataFrame, as_of=None, window: int = 4, top_k: int = 3,
                  n_evidence: int = 3) -> list[dict]:
    """Score every topic and return the top_k problems (score > 0), highest first."""
    if labeled.empty:
        return []
    as_of = pd.Timestamp(as_of) if as_of is not None else labeled["week"].max()
    start = as_of - pd.Timedelta(weeks=window - 1)
    prev_start = start - pd.Timedelta(weeks=window)
    recent = labeled[(labeled["week"] >= start) & (labeled["week"] <= as_of)]
    previous = labeled[(labeled["week"] >= prev_start) & (labeled["week"] < start)]

    overall_stars = recent.drop_duplicates("review_id")["stars"].mean()
    problems = []
    for topic in PROBLEM_TOPICS:
        cur = recent[recent["topic"] == topic]
        if cur.empty:
            continue
        mentions = len(cur)
        negativity = float(cur["sentiment"].map(NEG_WEIGHT).mean())
        neg_now = int((cur["sentiment"] == "negative").sum())
        neg_prev = int(((previous["topic"] == topic) & (previous["sentiment"] == "negative")).sum())
        trend = min(max((neg_now - neg_prev) / max(neg_prev, 1), TREND_CLIP[0]), TREND_CLIP[1])
        score = mentions * negativity * (1 + trend)
        if score <= 0:
            continue

        neg = cur[cur["sentiment"] == "negative"].sort_values("date", ascending=False)
        neg_reviews = neg.drop_duplicates("review_id")
        evidence = [
            {"review_id": r.review_id, "date": r.date.date().isoformat(), "stars": r.stars,
             "sentence": r.sentence, "detail": r.detail}
            for r in neg_reviews.head(n_evidence).itertuples(index=False)
        ]
        problems.append({
            "topic": topic,
            "score": round(score, 2),
            "mentions": mentions,
            "negative_mentions": neg_now,
            "negative_mentions_prev_window": neg_prev,
            "negativity": round(negativity, 2),
            "trend": round(trend, 2),
            "reviews_affected": int(neg_reviews["review_id"].nunique()),
            # Stars of reviews that complain about this topic vs. all reviews in the window: a simple
            # "likely rating impact" signal shown next to the score (not part of the score itself).
            "avg_stars_when_negative": round(float(neg_reviews["stars"].mean()), 2) if len(neg_reviews) else None,
            "avg_stars_overall": round(float(overall_stars), 2) if pd.notna(overall_stars) else None,
            "evidence": evidence,
        })
    problems.sort(key=lambda p: p["score"], reverse=True)
    return problems[:top_k]


def window_summary(labeled: pd.DataFrame, as_of=None, window: int = 4) -> dict:
    """Headline numbers for the dashboard and the brief."""
    as_of = pd.Timestamp(as_of) if as_of is not None else labeled["week"].max()
    start = as_of - pd.Timedelta(weeks=window - 1)
    recent = labeled[(labeled["week"] >= start) & (labeled["week"] <= as_of)]
    reviews = recent.drop_duplicates("review_id")
    return {
        "window_start": start.date().isoformat(),
        "window_end": (as_of + pd.Timedelta(days=6)).date().isoformat(),
        "weeks": window,
        "reviews": int(len(reviews)),
        "sentences": int(len(recent)),
        "avg_stars": round(float(reviews["stars"].mean()), 2) if reviews["stars"].notna().any() else None,
    }
