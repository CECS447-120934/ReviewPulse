import json
import os
import re

import llm
from schema import NONE_LABEL, SENTIMENTS, TOPICS
from store import LabelCache

_TOPIC_WORDS = {
    "wait_time": ["wait", "waited", "waiting", "slow", "minutes", "hour", "forever", "took ages", "quick", "fast", "prompt"],
    "service": ["server", "waiter", "waitress", "staff", "service", "host", "hostess", "manager", "rude", "friendly", "attentive", "bartender"],
    "food": ["food", "dish", "pasta", "steak", "burger", "pizza", "soup", "salad", "dessert", "flavor", "taste", "delicious", "bland", "cold", "fresh", "menu", "chicken", "fish", "tacos", "fries"],
    "price": ["price", "prices", "expensive", "cheap", "overpriced", "pricey", "value", "cost", "bill", "worth", "affordable"],
    "cleanliness": ["clean", "dirty", "filthy", "sticky", "bathroom", "restroom", "tables", "sanitary", "smelled", "greasy"],
    "ambience": ["ambience", "ambiance", "atmosphere", "music", "loud", "noisy", "decor", "cozy", "vibe", "lighting", "patio", "crowded"],
}
_NEG = {"slow", "rude", "bland", "cold", "dirty", "filthy", "sticky", "overpriced", "expensive", "pricey", "loud", "noisy",
        "forever", "ignored", "terrible", "awful", "bad", "worst", "disappointing", "disappointed", "greasy", "stale",
        "overcooked", "undercooked", "crowded", "mediocre", "poor", "horrible", "long", "never", "lukewarm", "late"}
_POS = {"delicious", "friendly", "attentive", "fast", "quick", "prompt", "clean", "cozy", "great", "excellent", "amazing",
        "fresh", "perfect", "love", "loved", "wonderful", "fantastic", "best", "affordable", "worth", "tasty", "lovely", "good"}
_NEGATORS = {"not", "no", "never", "wasn't", "wasnt", "isn't", "isnt", "didn't", "didnt", "don't", "dont", "hardly", "without"}


class KeywordLabeler:
    """Rule-based baseline labeler (offline, free)."""
    name = "keyword-rules"

    def label(self, sentences: list[str]) -> list[dict]:
        """Label each sentence; returns one dict per sentence."""
        return [self._one(s) for s in sentences]

    def _one(self, sentence: str) -> dict:
        """Label a single sentence with keyword rules."""
        low = sentence.lower()
        tokens = re.findall(r"[a-z']+", low)
        scores = {t: sum(1 for w in ws if w in low) for t, ws in _TOPIC_WORDS.items()}
        topic, best = max(scores.items(), key=lambda kv: kv[1])
        if best == 0:
            return dict(NONE_LABEL)
        pos = neg = 0
        for i, tok in enumerate(tokens):
            if tok in _NEG or tok in _POS:
                negated = any(t in _NEGATORS for t in tokens[max(0, i - 3):i])
                is_neg = (tok in _NEG) != negated
                neg += is_neg
                pos += not is_neg
        sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
        m = re.search(r"\b(\d+)[- ]?(minute|min|hour|hr)s?\b", low)
        return {"topic": topic, "sentiment": sentiment, "detail": m.group(0) if m else ""}

SYSTEM_PROMPT = """You label sentences from restaurant reviews. For EACH numbered sentence return one JSON object:
{"i": <number>, "topic": <one of food, service, wait_time, price, cleanliness, ambience, none>,
 "sentiment": <positive, neutral, negative>, "detail": <short phrase copied or paraphrased from the sentence, or "">}

Rules:
- topic is the main thing the sentence judges. Use "none" if it is not about the restaurant experience.
- Read negation and implied topics carefully: "we sat 45 minutes before anyone came over" is wait_time, negative, detail "45-minute wait".
- "The wait was not bad" is wait_time, positive.
- Sentiment is about the restaurant, not the reviewer's mood.
- Output ONLY a JSON array, one object per input sentence, in order. No commentary.

Examples:
Input: [{"i":1,"s":"Our server disappeared for half an hour."},{"i":2,"s":"The pasta was cooked perfectly."},{"i":3,"s":"We went here for my birthday."}]
Output: [{"i":1,"topic":"wait_time","sentiment":"negative","detail":"server absent 30 min"},{"i":2,"topic":"food","sentiment":"positive","detail":"pasta cooked perfectly"},{"i":3,"topic":"none","sentiment":"neutral","detail":""}]"""


def parse_llm_json(text: str, n: int) -> list[dict]:
    """Parse the model's reply into n label dicts; anything invalid becomes a neutral 'none'."""
    out = [dict(NONE_LABEL) for _ in range(n)]
    try:
        start, end = text.index("["), text.rindex("]") + 1
        items = json.loads(text[start:end])
    except (ValueError, json.JSONDecodeError):
        return out
    for it in items:
        try:
            i = int(it["i"]) - 1
            topic = it.get("topic", "none")
            sent = it.get("sentiment", "neutral")
            if 0 <= i < n:
                out[i] = {
                    "topic": topic if topic in TOPICS else "none",
                    "sentiment": sent if sent in SENTIMENTS else "neutral",
                    "detail": str(it.get("detail", ""))[:80],
                }
        except (KeyError, TypeError, ValueError):
            continue
    return out


class LLMLabeler:
    """Claude-based labeler with batching and caching."""
    def __init__(self, api_key: str | None = None, model: str | None = None, cache: LabelCache | None = None,
                 batch_size: int = 25, client=None):
        """Set up the API client, model name and cache. `client` lets tests pass a fake client."""
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.client = client
        self.model = model or llm.model_name()
        self.cache = cache
        self.batch_size = batch_size
        self.name = f"llm:{self.model}"
        self.api_calls = 0
        self.failed_batches = 0
        self.last_error = ""  # shown to the user so a silent keyword fallback can't pass as LLM output

    def label(self, sentences: list[str]) -> list[dict]:
        """Label each sentence; returns one dict per sentence."""
        results: list[dict | None] = [None] * len(sentences)
        todo = []
        for idx, s in enumerate(sentences):
            hit = self.cache.get(self.model, s) if self.cache is not None else None
            if hit:
                results[idx] = hit
            else:
                todo.append(idx)
        for b in range(0, len(todo), self.batch_size):
            chunk = todo[b:b + self.batch_size]
            labels = self._call([sentences[i] for i in chunk])
            if labels is None:  # API error or refusal: keyword labels, and don't cache them as LLM labels
                self.failed_batches += 1
                labels = KeywordLabeler().label([sentences[i] for i in chunk])
                for i, lab in zip(chunk, labels):
                    results[i] = lab
                continue
            for i, lab in zip(chunk, labels):
                results[i] = lab
                if self.cache is not None:
                    self.cache.put(self.model, sentences[i], lab)
        return results

    def _call(self, batch: list[str]) -> list[dict] | None:
        """Send one batch of sentences to Claude and parse the reply (None if the call failed)."""
        import anthropic

        payload = json.dumps([{"i": k + 1, "s": s} for k, s in enumerate(batch)])
        try:
            text = llm.complete(self.client, SYSTEM_PROMPT, f"Input: {payload}\nOutput:",
                                max_tokens=8000, effort="low", model=self.model)
        except (anthropic.APIError, llm.LLMRefusal) as e:
            self.last_error = f"{type(e).__name__}: {e}"[:200]
            return None
        finally:
            self.api_calls += 1
        return parse_llm_json(text, len(batch))


def get_labeler(prefer_llm: bool = True, cache: LabelCache | None = None):
    """LLM labeler if a key is configured, otherwise the keyword baseline."""
    if prefer_llm and os.environ.get("ANTHROPIC_API_KEY"):
        return LLMLabeler(cache=cache)
    return KeywordLabeler()
