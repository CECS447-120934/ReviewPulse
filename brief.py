"""Weekly brief: turn the ranked problems into a short note the owner reads in five minutes.

Grounding: the LLM only sees the ranked problems (numbers + quoted sentences), not raw reviews, and
is told to quote evidence verbatim. check_grounding() then verifies every quote in the brief really
appears in the evidence, so an invented quote is flagged to the owner instead of trusted.
"""
import json
import re

import llm

BRIEF_SYSTEM = """You write a weekly review brief for the owner of an independent restaurant.
You receive JSON with the review window and up to three ranked problems. Each problem has counts,
a trend versus the previous window, and evidence sentences quoted from real reviews.

Write in Markdown, under 220 words:
- One opening line: the window, number of reviews and average stars.
- One numbered item per problem, in the given order. For each: a bold headline with the topic and its
  negative-mention count and change versus the previous window; one quote copied EXACTLY, character for
  character, from that problem's evidence, in double quotes; one concrete, low-cost thing the owner could
  check or try this week, phrased as a suggestion.
Rules: use only numbers and quotes from the JSON. Do not invent causes, staff names, dates or events.
Do not promise anything to customers. No preamble, no sign-off."""

_QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,})[\"”]")
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", s).strip().lower().rstrip(".!?")


def check_grounding(text: str, problems: list[dict]) -> list[str]:
    """Quotes in the brief that do not appear in any evidence sentence."""
    evidence = [_norm(e["sentence"]) for p in problems for e in p["evidence"]]
    return [q for q in _QUOTE_RE.findall(text) if not any(_norm(q) in ev for ev in evidence)]


def _label(topic: str) -> str:
    return topic.replace("_", " ")


def _change(p: dict) -> str:
    diff = p["negative_mentions"] - p["negative_mentions_prev_window"]
    return f"up from {p['negative_mentions_prev_window']}" if diff > 0 else \
        f"down from {p['negative_mentions_prev_window']}" if diff < 0 else "unchanged"


def template_brief(summary: dict, problems: list[dict]) -> str:
    """Deterministic brief (no API key, or the LLM failed). Same facts, plainer wording."""
    stars = f", average {summary['avg_stars']} stars" if summary.get("avg_stars") is not None else ""
    lines = [f"**{summary['window_start']} to {summary['window_end']}:** {summary['reviews']} reviews{stars}.", ""]
    if not problems:
        lines.append("No recurring problems found in this window.")
    for i, p in enumerate(problems, 1):
        quote = p["evidence"][0]["sentence"] if p["evidence"] else ""
        n = p["negative_mentions"]
        lines.append(f"{i}. **{_label(p['topic']).capitalize()}: {n} negative mention{'s' if n != 1 else ''} "
                     f"({_change(p)} in the previous window).** \"{quote}\"")
    return "\n".join(lines)


def write_brief(summary: dict, problems: list[dict], client=None) -> dict:
    """Return {"text", "source", "unsupported_quotes"}."""
    text, source = None, "template"
    if client is not None and problems:
        import anthropic

        payload = {"window": summary, "problems": [
            {k: p[k] for k in ("topic", "negative_mentions", "negative_mentions_prev_window", "mentions",
                               "reviews_affected", "avg_stars_when_negative", "evidence")}
            for p in problems]}
        try:
            text = llm.complete(client, BRIEF_SYSTEM, json.dumps(payload, default=str),
                                max_tokens=4000, effort="medium")
            source = f"llm:{llm.model_name()}"
        except (anthropic.APIError, llm.LLMRefusal):
            text = None
    if not text:
        text, source = template_brief(summary, problems), "template"
    return {"text": text, "source": source, "unsupported_quotes": check_grounding(text, problems)}
