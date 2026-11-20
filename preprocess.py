"""Preprocess reviews: clean text, remove names/IDs/contact info, split into sentences.

to_sentences(reviews) turns the loader's review table into one row per sentence, ready for
LLMLabeler.label(df.sentence.tolist()) and weekly ranking.
"""
import html
import re
import sys
import unicodedata

import pandas as pd

MIN_WORDS = 3  # shorter fragments ("Wow!", "5 stars.") carry no labelable judgment

# --- cleaning ---------------------------------------------------------------
_QUOTES = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": " - "})
_SPACES_RE = re.compile(r"[ \t ]+")
_NEWLINES_RE = re.compile(r"\s*\n\s*")

# --- scrubbing (order matters: URLs/emails before phones/IDs) ----------------
_URL_RE = re.compile(r"\b(?:https?://|www\.)\S+", re.I)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\w)")
# Yelp IDs are 22 chars of [A-Za-z0-9_-]; require a digit so ordinary long words aren't hit.
_YELP_ID_RE = re.compile(r"(?<![\w-])(?=[A-Za-z0-9_-]*\d)[A-Za-z0-9_-]{22}(?![\w-])")

_ROLES = r"(?:server|waiter|waitress|bartender|host|hostess|manager|owner|chef|cook|busboy|busser|cashier|sommelier)"
# Case-sensitive even inside re.I patterns, so only capitalized words count: "Jessica", "Jessica M", "Jessica M."
_NAME = r"(?-i:[A-Z][a-z]+(?:\s[A-Z]\b\.?)?)"
_NOT_NAMES = {"The", "This", "That", "She", "He", "They", "It", "We", "Our", "Who", "And", "But", "Was", "Is",
              "Very", "Super", "So", "Too", "Also", "Then", "When", "Here", "There", "Thank", "Thanks"}
# "our server Jessica", "the manager, Tom,"
_ROLE_NAME_RE = re.compile(rf"(\b{_ROLES}s?,?\s+(?:named\s+|was\s+)?)({_NAME})\b", re.I)
# "Jessica was our waitress", "Tom (the manager)", "Tom, our server"
_NAME_ROLE_RE = re.compile(rf"\b({_NAME})(\s*(?:,|\(|was|is)\s*(?:our|the|my)\s+{_ROLES}\b)", re.I)
# "ask for Mike", "shoutout to Mike", "thanks to Mike"
_ASK_FOR_RE = re.compile(rf"(\b(?:ask\s+for|shout\s*-?\s*out\s+to|thanks?\s+to|thank\s+you)\s+)({_NAME})\b", re.I)
_HONORIFIC_RE = re.compile(r"\b(?:Mr|Mrs|Ms|Miss|Dr)\.?\s+[A-Z][a-z]+")

# --- sentence splitting -------------------------------------------------------
_ABBREV = ["Mr", "Mrs", "Ms", "Dr", "St", "Jr", "Sr", "vs", "etc", "approx", "min", "hr", "e.g", "i.e", "a.m", "p.m", "No"]
_ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREV) + r")\.", re.I)
_DECIMAL_RE = re.compile(r"(\d)\.(\d)")
_ELLIPSIS_RE = re.compile(r"\.{3,}|…")
_DOT = "․"  # placeholder for protected periods (one-dot leader), restored after splitting
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])[\"')\]]*\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def clean_text(s: str) -> str:
    """Normalize unicode, decode HTML entities, straighten quotes, and collapse whitespace (newlines kept)."""
    s = unicodedata.normalize("NFKC", html.unescape(str(s)))
    s = s.translate(_QUOTES)
    s = _SPACES_RE.sub(" ", s)
    return _NEWLINES_RE.sub("\n", s).strip()


def _keep_if_common_word(m: re.Match, name_group: int, repl: str) -> str:
    """Replace the name group with [NAME] unless it's a common capitalized word."""
    name = m.group(name_group)
    if name.split()[0] in _NOT_NAMES:
        return m.group(0)
    return m.group(0).replace(name, repl, 1)


def scrub(s: str) -> str:
    """Replace URLs, emails, phone numbers, Yelp IDs and staff/customer names with placeholders."""
    s = _URL_RE.sub("[URL]", s)
    s = _EMAIL_RE.sub("[EMAIL]", s)
    s = _PHONE_RE.sub("[PHONE]", s)
    s = _YELP_ID_RE.sub("[ID]", s)
    s = _HONORIFIC_RE.sub("[NAME]", s)
    s = _ROLE_NAME_RE.sub(lambda m: _keep_if_common_word(m, 2, "[NAME]"), s)
    s = _NAME_ROLE_RE.sub(lambda m: _keep_if_common_word(m, 1, "[NAME]"), s)
    s = _ASK_FOR_RE.sub(lambda m: _keep_if_common_word(m, 2, "[NAME]"), s)
    return s


def split_sentences(s: str, min_words: int = MIN_WORDS) -> list[str]:
    """Split on . ! ? and newlines, protecting abbreviations, decimals ($12.50) and ellipses."""
    protected = _ELLIPSIS_RE.sub(_DOT * 3, s)
    protected = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, protected)
    protected = _DECIMAL_RE.sub(rf"\1{_DOT}\2", protected)
    out = []
    for part in _SENT_SPLIT_RE.split(protected):
        sent = part.replace(_DOT, ".").strip()
        if len(_WORD_RE.findall(sent)) >= min_words:
            out.append(sent)
    return out


def preprocess_review(text: str) -> list[str]:
    """clean -> scrub -> split for one review."""
    return split_sentences(scrub(clean_text(text)))


def to_sentences(reviews: pd.DataFrame) -> pd.DataFrame:
    """One row per sentence: review_id, business_id, stars, date, week (Monday), sent_idx, sentence."""
    rows = []
    for r in reviews.itertuples(index=False):
        for i, sent in enumerate(preprocess_review(r.text)):
            rows.append((r.review_id, r.business_id, r.stars, r.date, i, sent))
    df = pd.DataFrame(rows, columns=["review_id", "business_id", "stars", "date", "sent_idx", "sentence"])
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = (df["date"] - pd.to_timedelta(df["date"].dt.day_of_week, unit="D")).dt.normalize()
    return df[["review_id", "business_id", "stars", "date", "week", "sent_idx", "sentence"]]


if __name__ == "__main__":
    from data import load_reviews

    if len(sys.argv) < 2:
        sys.exit("usage: python preprocess.py <reviews.csv | yelp data dir> [business_id]")
    kw = {"business_ids": sys.argv[2]} if len(sys.argv) > 2 else {}
    reviews = load_reviews(sys.argv[1], **kw)
    sents = to_sentences(reviews)
    print(f"{len(reviews)} reviews -> {len(sents)} sentences over {sents['week'].nunique()} weeks")
    with pd.option_context("display.max_colwidth", 90, "display.width", 160):
        print(sents[["week", "stars", "sentence"]].head(15).to_string(index=False))
