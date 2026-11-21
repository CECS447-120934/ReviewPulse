# Labeling guideline

Used for every human-labeled gold set in ReviewPulse: the sample sentences, the ~300 Yelp sentences, and any re-checks. Both labelers read this before starting and **label on their own, without looking at each other's sheet or the AI-drafted labels.** The Claude labeler's prompt (`labelers.py`, `SYSTEM_PROMPT`) follows the same rules.

## What to fill in

Each row is one sentence from a review. Give it exactly **one topic** and **one sentiment**. Use lowercase labels spelled exactly as below.

### Topic: the main thing the sentence judges

| Topic | Use for | Examples |
| --- | --- | --- |
| `food` | taste, freshness, temperature, portions, menu, drinks | "The tiramisu was perfect." · "Portions were tiny." |
| `service` | how staff behave: friendliness, attentiveness, mistakes, attitude | "Our server was friendly." · "The host was rude." |
| `wait_time` | how long anything took: table, order, food, check; slowness | "Mains took 45 minutes." · "Slow service again." |
| `price` | cost, value, the bill | "A bit pricey for what you get." |
| `cleanliness` | dirty or clean tables, floors, restrooms, dishes | "The bathroom was filthy." |
| `ambience` | noise, music, decor, lighting, seating comfort, crowding | "Way too loud to talk." |
| `none` | not a judgment of the restaurant: background, plans, the reviewer's own story | "We came for my birthday." |

### Sentiment: how the sentence judges that topic

- `positive`: praise, or something described as acceptable ("the wait was fine").
- `negative`: a complaint, even a mild one ("a bit slow").
- `neutral`: a fact with no judgment, or a judgment that is truly balanced ("nothing special"). Every `none` sentence is `neutral`.

## Rules for hard cases

1. **Slowness is `wait_time`, not `service`.** If the complaint is about *time*, use `wait_time`, even when the sentence says "service": "Service was slow" → `wait_time`, `negative`. Use `service` only for *how* staff behaved: "The host was rude when we asked about our wait" → `service`, `negative`. Phase 2 found this was the most common inconsistency, so check it twice.
2. **Read negation.** "The wait was not bad" → `wait_time`, `positive`. "Not the friendliest staff" → `service`, `negative`.
3. **Implied topics count.** "We sat 45 minutes before anyone came over" names no topic word but is `wait_time`, `negative`. "The tiramisu was perfect" is `food`.
4. **An acceptable wait is `positive`.** "Waited about 20 min, which is fine for a Thursday" → `wait_time`, `positive`. A wait mentioned with no judgment is `neutral`.
5. **Mixed sentences: pick the part the reviewer stresses**, usually the clause after "but". "Delicious, but the wait was annoying" → `wait_time`, `negative`. "Food was good but a bit pricey" → `price`, `negative`. If one sentence really covers two topics equally, still pick one, and write the other in a `note` column if you want; we keep one label per sentence (a known limitation).
6. **Judge the restaurant, not the mood.** "I was starving when we arrived" is `none`, `neutral`.
7. **Sarcasm:** label what is meant. "Great, another 40-minute wait" → `wait_time`, `negative`.
8. **Don't use the star rating** to decide a sentence's label. Judge the sentence by itself.

## How to label (step by step)

1. Copy the blind sheet, for example `data/gold/annotations/sample_blind.csv`, to `data/gold/annotations/sample_<yourname>.csv`. Don't reorder or edit sentences.
2. Fill in `topic` and `sentiment` for every row. Spreadsheet software is fine; save as CSV.
3. Run `python agreement.py compare <your file> <other labeler's file>`. It prints Cohen's kappa and writes `disagreements.csv`.
4. Meet and go through each disagreement. Fill in `final_topic` and `final_sentiment`. If a rule above was unclear, update this guideline.
5. Run `python agreement.py merge <A> <B> <disagreements.csv> data/gold/<set>_gold_human.csv`, then score the labelers with `python evaluate.py --gold data/gold/<set>_gold_human.csv`.

Report the kappa before resolving disagreements. That number shows how clear the task is for people; the resolved file is what the labelers are scored against. Kappa above 0.6 counts as substantial agreement (Landis & Koch, 1977).
