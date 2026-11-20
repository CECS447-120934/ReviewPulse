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

The Yelp Open Dataset is not included (its terms restrict redistribution). Download it from https://business.yelp.com/data/resources/open-dataset/ and place the files in `data/raw/`. TODO: sample CSV for a quick run.

## Run the prototype

TODO

## Reproduce a sample result

TODO

## Main files

TODO

## LLM prompts and API workflow

TODO: prompts, input/output format, limitations, and what the team built around the API.

## Completed vs. planned

| Feature | Status |
| --- | --- |
| TODO | |

## Known limitations

TODO
