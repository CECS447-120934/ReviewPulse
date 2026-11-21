"""Label vocabulary shared by the labelers, ranking, evaluation and dashboard."""

TOPICS = ["food", "service", "wait_time", "price", "cleanliness", "ambience", "none"]
SENTIMENTS = ["positive", "neutral", "negative"]

# Topics that can become a "problem" in the brief ("none" is never ranked).
PROBLEM_TOPICS = [t for t in TOPICS if t != "none"]

NONE_LABEL = {"topic": "none", "sentiment": "neutral", "detail": ""}
