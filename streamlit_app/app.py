"""Streamlit app for the sentiment classifier (Task 4).

The model is NOT stored in this GitHub repo. It is hosted on Hugging Face and
downloaded when the app starts (then cached by Streamlit).

Run locally:   streamlit run streamlit_app/app.py
Deploy:        Streamlit Community Cloud -> main file path: streamlit_app/app.py
"""
import streamlit as st
from transformers import pipeline

# Our Hugging Face model repository (a copy of cardiffnlp/twitter-roberta-base-sentiment-latest)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

LABELS = {
    "negative": ("😞", "Negative", "#FF7A7A"),
    "neutral": ("😐", "Neutral", "#9AA6FF"),
    "positive": ("😊", "Positive", "#3DDC97"),
}

EXAMPLES = [
    "Great tablet for the price. My kids love it and the battery lasts all day.",
    "It works, but the screen is dimmer than I expected for the price.",
    "Worst customer service I have ever experienced.",
]


@st.cache_resource(show_spinner="Loading the model from Hugging Face (first run only)...")
def load_model():
    """Download the model once and keep it in memory for every visitor."""
    return pipeline("text-classification", model=MODEL_NAME, top_k=None)


def predict(classifier, review):
    """Return {label: probability}, highest first."""
    out = classifier(review, truncation=True, max_length=512)
    out = out[0] if isinstance(out[0], list) else out
    scores = {d["label"].lower(): float(d["score"]) for d in out}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


# ---------------------------------------------------------------- page
st.set_page_config(page_title="Amazon Review Sentiment", page_icon="⭐", layout="centered")
st.title("⭐ Amazon Review Sentiment")
st.caption(f"Paste a product review and get negative / neutral / positive. Model: `{MODEL_NAME}` on Hugging Face.")

classifier = load_model()

example = st.selectbox("Try an example (optional)", ["—"] + EXAMPLES)
review = st.text_area(
    "Customer review",
    value="" if example == "—" else example,
    height=150,
    placeholder="Paste a product review here...",
)

if st.button("Analyse sentiment", type="primary"):
    if not review.strip():
        st.warning("Please enter a review first.")
    else:
        scores = predict(classifier, review.strip())
        top = next(iter(scores))
        emoji, name, color = LABELS.get(top, ("", top, "#999999"))

        st.markdown(
            f"<h2 style='color:{color};margin-bottom:0'>{emoji} {name}</h2>"
            f"<p style='margin-top:0'>confidence {scores[top]:.0%}</p>",
            unsafe_allow_html=True,
        )
        for label, p in scores.items():
            e, n, _ = LABELS.get(label, ("", label, ""))
            st.progress(p, text=f"{e} {n}: {p:.0%}")

st.divider()
st.caption("Ironhack AI Engineering · Project 4 · Hamza Shabbir & Yasaman Najafi Jozani")
