"""Inter-annotator agreement for the gold labels (Cohen's kappa), plus the blind-sheet and merge steps.

Workflow (see docs/labeling_guideline.md):

    # 1. Make a blank sheet: same sentences, labels removed, so labelers can't see the AI draft
    python agreement.py blind data/gold/sample_gold.csv data/gold/annotations/sample_blind.csv

    # 2. Two people each copy the sheet and fill in topic and sentiment on their own
    #    e.g. data/gold/annotations/sample_nathan.csv and data/gold/annotations/sample_<teammate>.csv

    # 3. Measure agreement; disagreements go to a file with blank final_topic / final_sentiment columns
    python agreement.py compare data/gold/annotations/sample_nathan.csv data/gold/annotations/sample_<teammate>.csv

    # 4. Discuss each disagreement, fill in final_topic / final_sentiment, then build the human gold set
    python agreement.py merge A.csv B.csv disagreements.csv data/gold/sample_gold_human.csv

The files must list the same sentences in the same order. Any CSV with sentence, topic and sentiment
columns works, so `compare` can also measure how far a human labeler is from the AI-drafted gold set.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from schema import SENTIMENTS, TOPICS

COLUMNS = ["sentence", "topic", "sentiment"]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two labelers over the same items: (p_observed - p_chance) / (1 - p_chance)."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of the same length")
    n = len(a)
    p_obs = sum(x == y for x, y in zip(a, b)) / n
    p_chance = sum((a.count(c) / n) * (b.count(c) / n) for c in set(a) | set(b))
    if p_chance == 1:  # both labelers used one identical label everywhere
        return 1.0
    return (p_obs - p_chance) / (1 - p_chance)


def interpret(kappa: float) -> str:
    """Landis & Koch (1977) bands, the usual way kappa is described in reports."""
    for limit, word in [(0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")]:
        if kappa <= limit:
            return word
    return "almost perfect"


def load_labels(path: str | Path) -> pd.DataFrame:
    """Read a filled-in sheet and check every row has a valid topic and sentiment."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {missing}")
    for col in COLUMNS:
        df[col] = df[col].str.strip()
    df["topic"] = df["topic"].str.lower()
    df["sentiment"] = df["sentiment"].str.lower()
    problems = []
    for i, row in df.iterrows():
        line = i + 2  # header is line 1
        if row["topic"] not in TOPICS:
            problems.append(f"line {line}: topic {row['topic']!r} (allowed: {', '.join(TOPICS)})")
        if row["sentiment"] not in SENTIMENTS:
            problems.append(f"line {line}: sentiment {row['sentiment']!r} (allowed: {', '.join(SENTIMENTS)})")
    if problems:
        shown = "\n  ".join(problems[:15]) + (f"\n  ... and {len(problems) - 15} more" if len(problems) > 15 else "")
        raise ValueError(f"{path}: {len(problems)} invalid or blank label(s):\n  {shown}")
    return df


def check_aligned(a: pd.DataFrame, b: pd.DataFrame) -> None:
    if len(a) != len(b):
        raise ValueError(f"files have different row counts ({len(a)} vs {len(b)})")
    for i, (x, y) in enumerate(zip(a["sentence"], b["sentence"])):
        if x != y:
            raise ValueError(f"line {i + 2}: sentences differ, rows must be in the same order:\n  {x!r}\n  {y!r}")


def compare(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agreement table (topic, sentiment, joint) and the rows where the labelers disagree."""
    check_aligned(a, b)
    joint_a = (a["topic"] + "/" + a["sentiment"]).tolist()
    joint_b = (b["topic"] + "/" + b["sentiment"]).tolist()
    rows = []
    for name, x, y in [("topic", a["topic"].tolist(), b["topic"].tolist()),
                       ("sentiment", a["sentiment"].tolist(), b["sentiment"].tolist()),
                       ("joint", joint_a, joint_b)]:
        k = cohen_kappa(x, y)
        rows.append({"label": name, "n": len(x),
                     "agreement": round(sum(p == q for p, q in zip(x, y)) / len(x), 3),
                     "kappa": round(k, 3), "strength": interpret(k)})
    differ = (a["topic"] != b["topic"]) | (a["sentiment"] != b["sentiment"])
    dis = pd.DataFrame({
        "row": a.index[differ] + 2,
        "sentence": a.loc[differ, "sentence"],
        "topic_a": a.loc[differ, "topic"], "topic_b": b.loc[differ, "topic"],
        "sentiment_a": a.loc[differ, "sentiment"], "sentiment_b": b.loc[differ, "sentiment"],
        "final_topic": "", "final_sentiment": "", "note": "",
    })
    return pd.DataFrame(rows), dis


def merge(a: pd.DataFrame, b: pd.DataFrame, resolved: pd.DataFrame) -> pd.DataFrame:
    """Gold set = rows both labelers agreed on + the final labels chosen for each disagreement."""
    check_aligned(a, b)
    gold = a[COLUMNS].copy()
    differ = (a["topic"] != b["topic"]) | (a["sentiment"] != b["sentiment"])
    resolved = resolved.fillna("")
    by_sentence = {str(r["sentence"]).strip(): r for _, r in resolved.iterrows()}
    problems = []
    for i in a.index[differ]:
        r = by_sentence.get(a.at[i, "sentence"])
        topic = str(r["final_topic"]).strip().lower() if r is not None else ""
        sent = str(r["final_sentiment"]).strip().lower() if r is not None else ""
        if topic not in TOPICS or sent not in SENTIMENTS:
            problems.append(f"line {i + 2}: {a.at[i, 'sentence']!r} -> final_topic={topic!r}, final_sentiment={sent!r}")
            continue
        gold.at[i, "topic"], gold.at[i, "sentiment"] = topic, sent
    if problems:
        raise ValueError(f"{len(problems)} disagreement(s) not resolved with valid labels:\n  " + "\n  ".join(problems))
    return gold


def make_blind(source: str | Path) -> pd.DataFrame:
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    return pd.DataFrame({"sentence": df["sentence"], "topic": "", "sentiment": ""})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("blind", help="copy sentences with empty label columns")
    p.add_argument("source")
    p.add_argument("out")
    p = sub.add_parser("compare", help="Cohen's kappa between two filled-in sheets")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("--out", default=None, help="disagreements CSV (default: next to file a)")
    p = sub.add_parser("merge", help="build the final gold set from two sheets and resolved disagreements")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("resolved")
    p.add_argument("out")
    args = ap.parse_args(argv)

    try:
        if args.cmd == "blind":
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            sheet = make_blind(args.source)
            sheet.to_csv(out, index=False)
            print(f"Wrote {out}: {len(sheet)} sentences, topic and sentiment left blank.")
        elif args.cmd == "compare":
            a, b = load_labels(args.a), load_labels(args.b)
            table, dis = compare(a, b)
            print(f"A = {args.a}\nB = {args.b}\n")
            print(table.to_string(index=False))
            print("\nTopic confusion (rows = A, columns = B):")
            print(pd.crosstab(a["topic"], b["topic"]).to_string())
            out = Path(args.out) if args.out else Path(args.a).with_name("disagreements.csv")
            dis.to_csv(out, index=False)
            print(f"\nWrote {out}: {len(dis)} disagreement(s). Fill in final_topic and final_sentiment, then run merge.")
        else:
            gold = merge(load_labels(args.a), load_labels(args.b), pd.read_csv(args.resolved, dtype=str))
            out = Path(args.out)
            gold.to_csv(out, index=False)
            print(f"Wrote {out}: {len(gold)} human-labeled sentences.")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
