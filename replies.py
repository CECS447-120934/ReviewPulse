"""Reply drafting. Always a draft: the owner edits and approves; nothing is posted."""
import re

import llm

REPLY_SYSTEM = """You write short, specific replies (2-4 sentences) from a restaurant owner to a customer review.
Rules: refer to something the reviewer actually said; thank them; if the review is negative, acknowledge the
specific issue without excuses; NEVER promise refunds, free items, discounts or compensation; NEVER invent facts
(no names, dates, policy changes, or actions the owner has not confirmed); do not copy a generic template.
Write only the reply text, in the owner's voice, no greeting placeholders."""

_PROMISE = re.compile(r"\b(refund|free|comp(ed|ensat\w*)?|discount|voucher|reimburs\w*|on the house|gift card)\b", re.I)


def promise_check(reply: str) -> list[str]:
    """Return risky words found; the app warns the owner before they copy the reply."""
    return sorted({m.group(0).lower() for m in _PROMISE.finditer(reply)})


def template_reply(stars: float, topics: list[str]) -> str:
    """Fallback when no API key is configured (generic by design, which is the weakness the LLM fixes)."""
    if stars >= 4:
        return "Thank you for the kind words and for dining with us. We hope to see you again soon."
    what = ", ".join(t.replace("_", " ") for t in topics) or "your visit"
    return f"Thank you for your feedback about {what}. We take it seriously and are looking into it."


def draft_reply(review_text: str, stars: float, topics: list[str], client=None, model: str | None = None) -> dict:
    """Return {"text", "source", "warnings"}. Uses the LLM when a client is given, else the template."""
    text = None
    if client is not None:
        import anthropic

        model = model or llm.model_name()
        try:
            # max_tokens leaves room for thinking (always on for Opus 5.5); the prompt keeps the reply short.
            text = llm.complete(client, REPLY_SYSTEM, f"Rating: {stars} stars\nReview: {review_text}",
                                max_tokens=2000, effort="low", model=model)
            source = f"llm:{model}"
        except (anthropic.APIError, llm.LLMRefusal):
            text = None
    if not text:
        text, source = template_reply(stars, topics), "template"
    return {"text": text, "source": source, "warnings": promise_check(text)}
