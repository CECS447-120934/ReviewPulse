# ReviewPulse: Prototype Progress Report

**CECS 451 Term Project, Phase 2 · Team 5:** Christian Vanegas, Nathan Sarkozy, Nguyen Nguyen, Nathan Fender
**Repository:** https://github.com/CECS447-120934/ReviewPulse

## 1. Objective and current scope

ReviewPulse gives independent restaurant owners a five-minute weekly brief of the three problems costing them stars, each linked to the reviews behind it, plus reply drafts they edit and approve. Phase 2 scope follows the Phase 1 plan: a PoC that labels one restaurant's reviews with an LLM and compares it to a baseline, extended to an MVP with ranking, problem cards, trend charts and reply drafts in a dashboard. Live platform connections, auto-posting, multi-location accounts and billing remain out of scope.

## 2. Completed features

- **End-to-end flow** in a Streamlit dashboard (`app.py`) and a CLI (`pipeline.py`): upload a CSV or use the sample → Analyze → KPIs, weekly trend chart, top-3 problem cards with quoted evidence, weekly brief → pick a review → draft, edit and copy a reply.
- **Data and privacy** (`data.py`, `preprocess.py`): loaders for Yelp JSON and any review CSV; text cleaning; scrubbing of staff and customer names, phones, emails, URLs and IDs *before* any LLM call; sentence splitting; weekly grouping.
- **AI labeling** (`labelers.py`): Claude Opus 5.5 labels each sentence with topic, sentiment and a short detail as JSON, in batches of 25, with a SQLite cache and an automatic keyword fallback.
- **Ranking and writing** (`ranking.py`, `brief.py`, `replies.py`): an explainable problem score, an LLM brief with a quote-grounding check, and reply drafts with a promise check.
- **Evaluation** (`evaluate.py`), 92 draft gold-labeled sentences, and 40 automated tests (no network needed).

## 3. Architecture

*[Insert the architecture diagram: the Mermaid chart in the README, rendered by GitHub, or the Phase 1 figure updated with file names.]*

Reviews (CSV/Yelp) → **Preprocess** (clean, scrub PII, split, week) → **LLM extraction** (topic, sentiment, detail; SQLite cache) → **Ranking** (no AI) → **LLM writing** (brief + grounding check; replies + promise check) → **Dashboard** → owner edits and approves. Each LLM step has an offline fallback (keyword rules, templates), so the prototype runs with or without an API key.

## 4. Data and model/algorithm

- **Data:** a synthetic sample of 38 reviews over 6 weeks with a planted, growing wait-time problem and fake PII; draft gold labels for its 92 sentences; Yelp Open Dataset loader ready (data kept out of the repo per Yelp's terms).
- **Model:** Claude Opus 5.5 (Anthropic API) with a structured few-shot prompt, low effort for labeling and medium for the brief, and server-side refusal fallback. The key is read from an environment variable.
- **Baseline:** keyword rules with negation handling.
- **Ranking:** score = mentions × negativity × (1 + trend). Negativity weights are negative 1, neutral 0.25 and positive 0. Trend compares negative mentions with the previous 4-week window, clipped to [−0.5, 2].

## 5. Preliminary results

On the sample (offline baseline, reproducible with `python evaluate.py --offline`):

| Labeler | Topic macro-F1 | Sentiment macro-F1 | Joint accuracy | Planted problem in top 3 |
| --- | --- | --- | --- | --- |
| Keyword rules | 0.789 | 0.464 | 0.457 | Yes (ranked 1st) |
| Claude Opus 5.5 | *TODO* | *TODO* | *TODO* | *TODO* |

The brief correctly ranks wait time first (5 negative mentions in the last 4 weeks, up from 0), followed by food and service, each with quoted evidence. Errors in the baseline show why an LLM is needed: implied topics ("The tiramisu was perfect." → no topic) and mixed sentences ("Food was good but a bit pricey." → neutral) are mislabeled. *TODO: Claude row, cost per 100 sentences, and one example LLM brief and reply after a run with the team's key.*

## 6. Known limitations

One label per sentence (mixed sentences lose a topic). The sample is small and synthetic, so trends are noisy. The gold labels are a single-annotator draft. The trend clip and weights are hand-picked. Regex name scrubbing misses some names. The grounding check covers quotes, not numbers. There is no fake-review filter yet, and English only.

## 7. Remaining tasks (Phase 3, due Dec 11)

1. Run the Claude labeler and brief on the sample and 2–3 Yelp restaurants; record F1, cost and time per 100 sentences.
2. Second labeler on the gold set (Cohen's kappa); add ~300 Yelp sentences and the SemEval-2014 restaurant set.
3. Add the pre-trained DeBERTa-v3 ABSA baseline.
4. Blind 1–5 ratings of LLM vs. template replies.
5. Failure analysis (including informal-English bias check); record the demo video; write the final report.

## 8. Member responsibilities

| Member | Current responsibility |
| --- | --- |
| Christian Vanegas | Repository and docs; ranking, brief, pipeline, dashboard, evaluation harness; Claude API integration |
| Nathan Fender | Data loading (Yelp, CSV), preprocessing and PII scrubbing, sample data, preprocessing tests |
| Nguyen Nguyen | LLM labeler and prompt, keyword baseline, reply drafting |
| Nathan Sarkozy | *TODO* |
