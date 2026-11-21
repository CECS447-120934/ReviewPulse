"""Offline evaluation: compare labelers against hand-labeled gold sentences.

    python evaluate.py                          # keyword baseline (+ Claude if ANTHROPIC_API_KEY is set)
    python evaluate.py --gold data/gold/sample_gold.csv --offline

Metrics per labeler: topic macro-F1, sentiment macro-F1, joint accuracy (topic and sentiment both
right), seconds and estimated API cost per 100 sentences. Also checks that the wait-time problem
built into the sample data reaches the top three. Mislabeled sentences go to outputs/eval_errors.csv
for failure analysis.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

import llm
from labelers import KeywordLabeler, LLMLabeler
from schema import SENTIMENTS, TOPICS
from store import LabelCache


def macro_f1(gold: list[str], pred: list[str], classes: list[str]) -> float:
    """Unweighted mean F1 over the classes that appear in gold or predictions."""
    present = [c for c in classes if c in set(gold) | set(pred)]
    scores = []
    for c in present:
        tp = sum(g == c and p == c for g, p in zip(gold, pred))
        fp = sum(g != c and p == c for g, p in zip(gold, pred))
        fn = sum(g == c and p != c for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def score(gold: pd.DataFrame, labels: list[dict]) -> dict:
    pt = [l["topic"] for l in labels]
    ps = [l["sentiment"] for l in labels]
    gt, gs = gold["topic"].tolist(), gold["sentiment"].tolist()
    return {
        "topic_macro_f1": round(macro_f1(gt, pt, TOPICS), 3),
        "sentiment_macro_f1": round(macro_f1(gs, ps, SENTIMENTS), 3),
        "joint_accuracy": round(sum(a == b and c == d for a, b, c, d in zip(gt, pt, gs, ps)) / len(gold), 3),
    }


def evaluate(gold: pd.DataFrame, labeler) -> tuple[dict, list[dict]]:
    llm.reset_usage()
    t0 = time.perf_counter()
    labels = labeler.label(gold["sentence"].tolist())
    secs = time.perf_counter() - t0
    row = {"labeler": labeler.name, "n": len(gold), **score(gold, labels),
           "sec_per_100": round(secs / len(gold) * 100, 2),
           "usd_per_100": round(llm.usage_cost() / len(gold) * 100, 4) if llm.USAGE["calls"] else 0.0,
           "failed_batches": getattr(labeler, "failed_batches", 0)}
    if row["failed_batches"]:
        # Some sentences got keyword labels: these numbers are NOT a valid LLM result.
        row["labeler"] += " (INVALID: API failures)"
        print(f"WARNING: {row['failed_batches']} batches failed ({labeler.last_error}). "
              "Fix the key/connection and re-run before reporting these numbers.\n")
    return row, labels


def top3_check(source: str = "data/sample_reviews.csv", labeler=None, topic: str = "wait_time") -> bool:
    """Does the planted problem (rising wait-time complaints in the sample) reach the top three?"""
    from data import load_reviews
    from pipeline import analyze

    result = analyze(load_reviews(source), labeler or KeywordLabeler())
    return topic in [p["topic"] for p in result.problems]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", default="data/gold/sample_gold.csv")
    ap.add_argument("--offline", action="store_true", help="skip the Claude labeler")
    ap.add_argument("--no-cache", action="store_true", help="re-label with Claude even if cached")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args(argv)

    gold = pd.read_csv(args.gold)
    labelers = [KeywordLabeler()]
    if not args.offline and llm.has_key():
        labelers.append(LLMLabeler(cache=None if args.no_cache else LabelCache()))
    elif not args.offline:
        print("ANTHROPIC_API_KEY not set: evaluating the keyword baseline only.\n")

    rows, errors = [], []
    for lab in labelers:
        row, labels = evaluate(gold, lab)
        row["wait_time_in_top3"] = top3_check(labeler=lab)
        rows.append(row)
        for (_, g), p in zip(gold.iterrows(), labels):
            if (g["topic"], g["sentiment"]) != (p["topic"], p["sentiment"]):
                errors.append({"labeler": lab.name, "sentence": g["sentence"], "gold_topic": g["topic"],
                               "pred_topic": p["topic"], "gold_sentiment": g["sentiment"],
                               "pred_sentiment": p["sentiment"]})

    table = pd.DataFrame(rows)
    print(f"Gold set: {args.gold} ({len(gold)} sentences)\n")
    print(table.to_string(index=False))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "eval_results.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "eval_errors.csv", index=False)
    print(f"\nWrote {out / 'eval_results.csv'} and {out / 'eval_errors.csv'} ({len(errors)} mislabeled rows)")


if __name__ == "__main__":
    main()
