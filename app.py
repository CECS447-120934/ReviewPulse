"""ReviewPulse dashboard (Streamlit).

    streamlit run app.py

Flow: choose the sample or upload a CSV -> Analyze -> trend chart and top-3 problem cards with quoted
evidence -> weekly brief -> pick a review, draft a reply, edit it, copy it. Nothing is ever posted.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

import llm
from brief import write_brief
from data import load_csv
from pipeline import analyze, make_labeler
from replies import draft_reply

SAMPLE = "data/sample_reviews.csv"

st.set_page_config(page_title="ReviewPulse", layout="wide")
st.title("ReviewPulse")
st.caption("The three things costing you stars, with the reviews behind them.")

with st.sidebar:
    st.header("Reviews")
    uploaded = st.file_uploader("Upload a review CSV", type="csv",
                                help="Needs a text column (text/review/comment) and a date column. "
                                     "A rating column is optional.")
    use_llm = st.toggle("Use Claude", value=llm.has_key(), disabled=not llm.has_key(),
                        help="Off, or no ANTHROPIC_API_KEY: offline keyword rules and template text.")
    window = st.slider("Weeks per comparison window", 2, 8, 4)
    run = st.button("Analyze", type="primary", width="stretch")
    st.caption(f"Model: {llm.model_name()}" if llm.has_key() else "No API key found: offline mode.")

if run:
    try:
        reviews = load_csv(uploaded) if uploaded is not None else load_csv(SAMPLE, business_id="sample")
        with st.spinner("Labeling sentences and ranking problems..."):
            llm.reset_usage()
            result = analyze(reviews, make_labeler(offline=not use_llm), window=window)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as e:
        st.error(f"Could not analyze this file: {e}")
        st.stop()
    with st.spinner("Writing the brief..."):
        client = llm.get_client() if use_llm else None
        st.session_state.update(
            reviews=reviews, result=result, client=client,
            brief=write_brief(result.summary, result.problems, client=client),
            usage=dict(llm.USAGE), source=uploaded.name if uploaded is not None else "sample data",
        )

if "result" not in st.session_state:
    st.info("Choose a source in the sidebar and press **Analyze**. Without an upload, the built-in sample "
            "restaurant (38 reviews over 6 weeks) is used.")
    st.stop()

reviews, result, client = st.session_state.reviews, st.session_state.result, st.session_state.client
summary, problems, brief = result.summary, result.problems, st.session_state.brief
usage = st.session_state.usage

c1, c2, c3, c4 = st.columns(4)
c1.metric("Reviews in window", summary["reviews"])
c2.metric("Average stars", summary["avg_stars"] if summary["avg_stars"] is not None else "n/a")
c3.metric("Sentences labeled", len(result.labeled))
c4.metric("Labeler", result.labeler_name.replace("llm:", ""))
cost = f", {usage['calls']} API calls, est. ${llm.usage_cost(usage):.3f}" if usage["calls"] else ""
st.caption(f"Source: {st.session_state.source}. Window {summary['window_start']} to {summary['window_end']}. "
           f"Labeling and ranking took {result.seconds}s{cost}.")
if result.fallback_note:
    st.warning(result.fallback_note)

st.subheader("Weekly brief")
st.markdown(brief["text"])
st.caption(f"Written by: {brief['source']}")
if brief["unsupported_quotes"]:
    st.warning("These quotes were not found in the evidence and may be invented: "
               + "; ".join(brief["unsupported_quotes"]))

st.subheader("Negative mentions per week")
weekly = result.weekly_negative.reset_index().melt(id_vars="week", var_name="topic", value_name="negative")
top_topics = [p["topic"] for p in problems] or list(result.weekly_negative.columns)
fig = px.line(weekly[weekly["topic"].isin(top_topics)], x="week", y="negative", color="topic", markers=True,
              labels={"week": "Week starting", "negative": "Negative sentences"})
fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), legend_title_text="")
st.plotly_chart(fig, width="stretch")

st.subheader("Top problems")
if not problems:
    st.success("No recurring problems in this window.")
cols = st.columns(max(len(problems), 1))
for col, (i, p) in zip(cols, enumerate(problems, 1)):
    with col, st.container(border=True):
        st.markdown(f"**{i}. {p['topic'].replace('_', ' ').capitalize()}**")
        st.metric("Negative mentions", p["negative_mentions"],
                  delta=p["negative_mentions"] - p["negative_mentions_prev_window"], delta_color="inverse")
        st.caption(f"Score {p['score']} = {p['mentions']} mentions x {p['negativity']} negativity x "
                   f"(1 + {p['trend']} trend). {p['reviews_affected']} reviews affected"
                   + (f", averaging {p['avg_stars_when_negative']} stars vs {p['avg_stars_overall']} overall."
                      if p["avg_stars_when_negative"] is not None else "."))
        for e in p["evidence"]:
            st.markdown(f"> {e['sentence']}  \n<small>{e['date']} · {e['stars']:g} stars</small>"
                        if pd.notna(e["stars"]) else f"> {e['sentence']}  \n<small>{e['date']}</small>",
                        unsafe_allow_html=True)

st.subheader("Draft a reply")
problem_reviews = {e["review_id"] for p in problems for e in p["evidence"]}
options = reviews.assign(flag=reviews["review_id"].isin(problem_reviews)).sort_values(
    ["flag", "date"], ascending=[False, False])


def _fmt(rid):
    r = options.set_index("review_id").loc[rid]
    stars = f"{r['stars']:g}★ " if pd.notna(r["stars"]) else ""
    return f"{'⚑ ' if r['flag'] else ''}{r['date'].date()} {stars}{r['text'][:70]}..."


rid = st.selectbox("Review (⚑ = evidence for a top problem)", options["review_id"], format_func=_fmt)
review = reviews.set_index("review_id").loc[rid]
st.markdown(f"> {review['text']}")
if st.button("Draft reply"):
    topics = result.labeled.loc[(result.labeled["review_id"] == rid) &
                                (result.labeled["topic"] != "none"), "topic"].unique().tolist()
    st.session_state.reply = draft_reply(review["text"], review["stars"], topics, client=client)
    st.session_state.reply_for = rid
if st.session_state.get("reply_for") == rid:
    reply = st.session_state.reply
    edited = st.text_area("Edit before you copy it (nothing is posted)", reply["text"], height=120)
    st.caption(f"Drafted by: {reply['source']}")
    from replies import promise_check

    if warnings := promise_check(edited):
        st.warning(f"This reply mentions {', '.join(warnings)}. Make sure you intend to offer that.")
    st.code(edited, language=None)

with st.expander("All labeled sentences"):
    st.dataframe(result.labeled[["date", "stars", "sentence", "topic", "sentiment", "detail"]],
                 width="stretch", hide_index=True)
