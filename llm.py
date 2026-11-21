"""One place for every Claude API call (labeling, brief, replies).

- The key comes from ANTHROPIC_API_KEY (environment or a local .env file); it is never hard-coded.
- Claude Opus 5.5 always thinks, so a response can start with a `thinking` block. complete() returns
  only the text blocks, never content[0] blindly.
- A refusal (stop_reason == "refusal") raises LLMRefusal so callers can fall back instead of
  silently treating an empty answer as a result.
- Token usage is tallied in USAGE so the pipeline and evaluation can report cost per run.
"""
import os

try:  # optional: load a local .env so `python app.py` works without exporting the key
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_MODEL = "claude-opus-5-5"
# USD per million tokens (Anthropic pricing for Claude Opus 5.5).
PRICE_PER_MTOK = {"claude-opus-5-5": (4.00, 20.00)}

USAGE = {"calls": 0, "input_tokens": 0, "output_tokens": 0}


class LLMRefusal(RuntimeError):
    pass


def model_name() -> str:
    return os.environ.get("REVIEWPULSE_MODEL", DEFAULT_MODEL)


def has_key() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def get_client():
    """Anthropic client, or None when no key is configured (callers then use offline fallbacks)."""
    if not has_key():
        return None
    import anthropic

    return anthropic.Anthropic()


def complete(client, system: str, user: str, max_tokens: int = 4096, effort: str = "low",
             model: str | None = None) -> str:
    """Send one request and return the concatenated text of the reply."""
    model = model or model_name()
    msg = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": effort},
        # If a safety classifier declines, the API retries on a fallback model in the same call.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    USAGE["calls"] += 1
    usage = getattr(msg, "usage", None)
    if usage is not None:
        USAGE["input_tokens"] += getattr(usage, "input_tokens", 0) or 0
        USAGE["output_tokens"] += getattr(usage, "output_tokens", 0) or 0
    if getattr(msg, "stop_reason", None) == "refusal":
        raise LLMRefusal("the model declined this request")
    return "".join(b.text for b in msg.content if getattr(b, "type", None) == "text").strip()


def usage_cost(usage: dict | None = None, model: str | None = None) -> float:
    """Dollar cost of the tokens tallied in `usage` (defaults to the running USAGE)."""
    usage = usage or USAGE
    p_in, p_out = PRICE_PER_MTOK.get(model or model_name(), PRICE_PER_MTOK[DEFAULT_MODEL])
    return usage["input_tokens"] / 1e6 * p_in + usage["output_tokens"] / 1e6 * p_out


def reset_usage() -> None:
    for k in USAGE:
        USAGE[k] = 0
