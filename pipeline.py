"""End-to-end run: load -> preprocess -> label -> rank -> brief.

    python pipeline.py data/sample_reviews.csv                 # Claude if ANTHROPIC_API_KEY is set
    python pipeline.py data/sample_reviews.csv --offline       # keyword baseline, no API calls
    python pipeline.py data/raw --business-id <id> --limit 400 # Yelp Open Dataset

Writes <out>/labels.csv, <out>/problems.json and <out>/brief.md.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import llm
from brief import write_brief
from labelers import KeywordLabeler, LLMLabeler
from preprocess import to_sentences
from ranking import attach_labels, rank_problems, weekly_counts, window_summary
from store import LabelCache


@dataclass
class Analysis:
    labeled: pd.DataFrame
    problems: list[dict]
    summary: dict
    weekly_negative: pd.DataFrame
    labeler_name: str
    seconds: float
    fallback_note: str = ""  # non-empty when some LLM batches failed and used keyword labels instead


def make_labeler(offline: bool = False, cache: LabelCache | None = None):
    if offline or not llm.has_key():
        return KeywordLabeler()
    return LLMLabeler(cache=cache or LabelCache())


def analyze(reviews: pd.DataFrame, labeler, as_of=None, window: int = 4) -> Analysis:
    t0 = time.perf_counter()
    sentences = to_sentences(reviews).reset_index(drop=True)
    if sentences.empty:
        raise ValueError(f"no usable review sentences ({len(reviews)} reviews loaded; sentences need at least "
                         "3 words and a valid date)")
    labels = labeler.label(sentences["sentence"].tolist())
    labeled = attach_labels(sentences, labels)
    return Analysis(
        labeled=labeled,
        problems=rank_problems(labeled, as_of=as_of, window=window),
        summary=window_summary(labeled, as_of=as_of, window=window),
        weekly_negative=weekly_counts(labeled),
        labeler_name=labeler.name,
        seconds=round(time.perf_counter() - t0, 2),
        fallback_note=fallback_note(labeler),
    )


def fallback_note(labeler) -> str:
    failed = getattr(labeler, "failed_batches", 0)
    if not failed:
        return ""
    return (f"{failed} of {labeler.api_calls} Claude batches failed and used keyword labels instead "
            f"(last error: {labeler.last_error})")


def main(argv=None):
    from data import load_reviews

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="review CSV or Yelp data directory")
    ap.add_argument("--business-id", help="Yelp business_id (Yelp source only)")
    ap.add_argument("--limit", type=int, help="max reviews to load (Yelp source only)")
    ap.add_argument("--offline", action="store_true", help="keyword baseline only; no API calls")
    ap.add_argument("--as-of", help="Monday of the last week to include (default: latest week in data)")
    ap.add_argument("--window", type=int, default=4, help="weeks per comparison window (default 4)")
    ap.add_argument("--out", default="outputs", help="output directory (default outputs/)")
    args = ap.parse_args(argv)

    kw = {}
    if args.business_id:
        kw["business_ids"] = args.business_id
    if args.limit:
        kw["limit"] = args.limit
    labeler = make_labeler(args.offline)
    llm.reset_usage()
    try:
        reviews = load_reviews(args.source, **kw)
        result = analyze(reviews, labeler, as_of=args.as_of, window=args.window)
    except (ValueError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}")
    client = None if args.offline else llm.get_client()
    brief = write_brief(result.summary, result.problems, client=client)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.labeled.to_csv(out / "labels.csv", index=False)
    (out / "problems.json").write_text(json.dumps(
        {"labeler": result.labeler_name, "summary": result.summary, "problems": result.problems},
        indent=2, default=str), encoding="utf-8")
    (out / "brief.md").write_text(brief["text"] + "\n", encoding="utf-8")

    print(f"{len(reviews)} reviews -> {len(result.labeled)} sentences | labeler: {result.labeler_name} | "
          f"brief: {brief['source']} | {result.seconds}s labeling+ranking")
    if result.fallback_note:
        print("WARNING:", result.fallback_note)
    if llm.USAGE["calls"]:
        print(f"API calls: {llm.USAGE['calls']}, tokens in/out: {llm.USAGE['input_tokens']}/"
              f"{llm.USAGE['output_tokens']}, est. cost ${llm.usage_cost():.4f}")
    print("\nTop problems:")
    for i, p in enumerate(result.problems, 1):
        print(f"  {i}. {p['topic']:<12} score {p['score']:>6}  negative {p['negative_mentions']} "
              f"(prev {p['negative_mentions_prev_window']})  trend {p['trend']:+}")
    print("\n" + brief["text"])
    if brief["unsupported_quotes"]:
        print("\nWARNING: quotes not found in evidence:", brief["unsupported_quotes"])
    print(f"\nWrote {out / 'labels.csv'}, {out / 'problems.json'}, {out / 'brief.md'}")


if __name__ == "__main__":
    main()
