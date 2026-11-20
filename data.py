"""Load reviews from the Yelp Open Dataset or a CSV into one common table.

Every loader returns a DataFrame with columns:
    review_id (str), business_id (str), stars (float), date (datetime64), text (str)
"""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

YELP_BUSINESS = "yelp_academic_dataset_business.json"
YELP_REVIEW = "yelp_academic_dataset_review.json"
COLUMNS = ["review_id", "business_id", "stars", "date", "text"]

# CSV header aliases, matched after normalizing the header (lowercase, spaces/dashes -> "_").
_ALIASES = {
    "text": re.compile(r"^(text|review|review_text|comment|comments|body|content)$"),
    "stars": re.compile(r"^(stars|star|rating|ratings|score|review_stars)$"),
    "date": re.compile(r"^(date|created_at|created|review_date|time|timestamp)$"),
    "business_id": re.compile(r"^(business_id|business|restaurant|restaurant_id|location)$"),
    "review_id": re.compile(r"^(review_id|id)$"),
}
_HEADER_NORM_RE = re.compile(r"[\s\-]+")


def _review_id(text: str, date) -> str:
    """Stable ID from text and date, so re-loading the same CSV gives the same IDs (and cache hits)."""
    return hashlib.sha1(f"{date}|{text}".encode("utf-8")).hexdigest()[:16]


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, drop unusable rows, and return the common column order."""
    df = df.copy()
    df["text"] = df["text"].astype("string").str.strip()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["stars"] = pd.to_numeric(df["stars"], errors="coerce").astype(float)
    df = df[df["text"].notna() & (df["text"] != "") & df["date"].notna()]
    df["text"] = df["text"].astype(str)
    df["business_id"] = df["business_id"].astype(str)
    df["review_id"] = df["review_id"].astype(str)
    return df[COLUMNS].sort_values("date", kind="stable").reset_index(drop=True)


def list_restaurants(raw_dir: str | Path = "data/raw", city: str | None = None, min_reviews: int = 50) -> pd.DataFrame:
    """Restaurants in the Yelp business file, most-reviewed first. Use it to pick a business_id."""
    rows = []
    city_l = city.lower() if city else None
    with open(Path(raw_dir) / YELP_BUSINESS, encoding="utf-8") as f:
        for line in f:
            b = json.loads(line)
            if "Restaurants" not in (b.get("categories") or ""):
                continue
            if (b.get("review_count") or 0) < min_reviews:
                continue
            if city_l and (b.get("city") or "").lower() != city_l:
                continue
            rows.append({k: b.get(k) for k in ("business_id", "name", "city", "state", "stars", "review_count")})
    cols = ["business_id", "name", "city", "state", "stars", "review_count"]
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("review_count", ascending=False).reset_index(drop=True)


def load_yelp(raw_dir: str | Path = "data/raw", business_ids: str | list[str] | None = None,
              limit: int | None = None, load_all: bool = False) -> pd.DataFrame:
    """Load Yelp reviews, optionally only for some businesses.

    load_all=False (default): stream the ~5 GB file line by line. A substring check on business_id
        runs before json.loads, so other businesses' lines are never parsed. Low memory; stops at `limit`.
    load_all=True: read the whole file with pandas' C JSON parser, then filter. Faster when you want
        many businesses or no filter, but needs roughly 10+ GB of RAM for the full file.
    """
    path = Path(raw_dir) / YELP_REVIEW
    if isinstance(business_ids, str):
        business_ids = [business_ids]
    wanted = set(business_ids) if business_ids else None

    if load_all:
        df = pd.read_json(path, lines=True, dtype={"business_id": str, "review_id": str})
        if wanted:
            df = df[df["business_id"].isin(wanted)]
        if limit is not None:
            df = df.head(limit)
        return _finalize(df)

    # Bare-ID substring check (whitespace-agnostic); json.loads below confirms it's the business_id field.
    needles = list(wanted) if wanted else None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if needles and not any(n in line for n in needles):
                continue
            r = json.loads(line)
            if wanted and r.get("business_id") not in wanted:
                continue
            rows.append({k: r.get(k) for k in COLUMNS})
            if limit is not None and len(rows) >= limit:
                break
    return _finalize(pd.DataFrame(rows, columns=COLUMNS))


def load_csv(path: str | Path, business_id: str = "csv") -> pd.DataFrame:
    """Load a review CSV with flexible headers. Only a text column is required.

    Missing business_id defaults to `business_id`; missing review_id is a hash of text+date;
    missing date makes every row unusable, so a date column is effectively required too.
    """
    raw = pd.read_csv(path)
    rename = {}
    for col in raw.columns:
        norm = _HEADER_NORM_RE.sub("_", str(col).strip().lower())
        for target, pattern in _ALIASES.items():
            if target not in rename.values() and pattern.match(norm):
                rename[col] = target
                break
    df = raw.rename(columns=rename)
    if "text" not in df.columns:
        raise ValueError(f"{path}: no review text column found (expected one of: text, review, comment, body). "
                         f"Columns were: {list(raw.columns)}")
    if "date" not in df.columns:
        raise ValueError(f"{path}: no date column found (expected one of: date, created_at, review_date).")
    if "stars" not in df.columns:
        df["stars"] = float("nan")
    if "business_id" not in df.columns:
        df["business_id"] = business_id
    if "review_id" not in df.columns:
        df["review_id"] = [_review_id(t, d) for t, d in zip(df["text"], df["date"])]
    return _finalize(df)


def load_reviews(source: str | Path, **kw) -> pd.DataFrame:
    """Load from a .csv file or a Yelp directory (kwargs go to load_csv / load_yelp)."""
    p = Path(source)
    if p.suffix.lower() == ".csv":
        return load_csv(p, **kw)
    if p.is_dir():
        return load_yelp(p, **kw)
    raise ValueError(f"Don't know how to load {source}: pass a .csv file or the Yelp data directory.")
