# ReviewPulse

AI review insights for independent restaurants. ReviewPulse turns a restaurant's reviews into a weekly brief of the three problems hurting its rating most, with trends and quoted evidence, and drafts replies the owner edits and approves.

CECS 451 Term Project, Team 5: Christian Vanegas, Nathan Sarkozy, Nguyen Nguyen, Nathan Fender

> Status: Phase 2 (prototype) in progress. Sections marked TODO are filled in as code lands.

## Architecture

Reviews (Yelp data or CSV) → Preprocess (clean, split sentences, remove names/IDs) → **LLM extraction** (topic, sentiment, detail as JSON) → Label store (SQLite, by week) → Ranking (mentions × negativity × (1 + trend), top 3) → **LLM writing** (weekly brief, reply drafts) → Dashboard (Streamlit + Plotly) → Owner reviews, edits and approves.

Offline evaluation compares the LLM extraction against keyword rules and a pre-trained ABSA model (macro-F1).

TODO: diagram image in `docs/`.

## Setup

Requirements: Python 3.10+ and an Anthropic API key.

```bash
git clone <repo-url>
cd ReviewPulse
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then put your ANTHROPIC_API_KEY in .env
```

## Data

The Yelp Open Dataset is not included (its terms restrict redistribution). Download it from https://business.yelp.com/data/resources/open-dataset/ and place the files in `data/raw/`. For a quick run without it, `data/sample_reviews.csv` has 38 synthetic reviews of a fictional restaurant over 6 weeks.

Loading and preprocessing (`data.py`, `preprocess.py`) need only `pandas`; tests need `pytest`. Both are in `requirements.txt`.

## Run the prototype

Preprocessing only, so far (load → clean → remove names/IDs → split into sentences by week):

```bash
python preprocess.py data/sample_reviews.csv
python preprocess.py data/raw <business_id>   # Yelp; find IDs with data.list_restaurants()
python -m pytest tests/ -q
```

TODO: labeling, ranking, brief and dashboard.

## Reproduce a sample result

TODO

## Main files

TODO

## LLM prompts and API workflow

1. Sentence labeling (`labelers.py`)
- `LLMLabeler` sends review sentences to Claude in batches of 25. Input is a JSON list of numbered sentences.
- `SYSTEM_PROMPT` sets the rules and gives one few-shot example. Claude returns a JSON array with `topic`
  (food, service, wait_time, price, cleanliness, ambience or none), `sentiment` (positive, neutral, negative) and a short `detail`.
- `parse_llm_json` checks the reply. Invalid JSON or labels outside the allowed lists become `none / neutral`.
- Labels are cached in SQLite, so each sentence is only sent once.
- `KeywordLabeler` is the offline baseline. `get_labeler()` uses Claude if `ANTHROPIC_API_KEY` is set, otherwise the keyword rules.

2. Reply drafting (`replies.py`)
- `draft_reply` sends one review and its star rating to Claude. `REPLY_SYSTEM` requires a short, specific reply that never
  promises refunds, free items or discounts and never invents facts.
- `promise_check` warns if a reply contains words like "refund" or "free". Without a key, a generic template reply is used.
- Replies are only drafts. The owner edits them, and nothing is posted.
## Completed vs. planned

| Feature | Status |
| --- | --- |
| TODO | |

## Known limitations

TODO
