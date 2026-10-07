"""Gradio demo for the sentiment classifier (Task 4).

Which model is served is decided in notebook 02 and written to model_config.json:
  {"type": "tfidf"}                              -> sentiment_model.joblib (TF-IDF + Logistic Regression)
  {"type": "transformer", "model_id": "..."}     -> pretrained Hugging Face model

Run locally:   cd app && python app.py   -> http://127.0.0.1:7860
On HF Spaces:  upload app.py, text_cleaning.py, requirements.txt, model_config.json
               (+ sentiment_model.joblib for the TF-IDF model)
"""
import json
from pathlib import Path

import gradio as gr

HERE = Path(__file__).parent
CONFIG = json.loads((HERE / "model_config.json").read_text())

if CONFIG["type"] == "transformer":
    from transformers import pipeline

    classifier = pipeline("text-classification", model=CONFIG["model_id"], top_k=None)
    MODEL_NAME = f"pretrained transformer ({CONFIG['model_id']})"

    def scores(review):
        out = classifier(review, truncation=True, max_length=512)
        out = out[0] if isinstance(out[0], list) else out
        return {d["label"].lower(): float(d["score"]) for d in out}
else:
    import joblib
    import text_cleaning  # noqa: F401  the saved Pipeline calls text_cleaning.preprocess

    model = joblib.load(HERE / "sentiment_model.joblib")
    MODEL_NAME = "TF-IDF + Logistic Regression"

    def scores(review):
        probs = model.predict_proba([review])[0]
        return {label: float(p) for label, p in zip(model.classes_, probs)}


def predict(review):
    """Return {label: probability} for one review, for the Gradio Label output."""
    review = (review or "").strip()
    return scores(review) if review else {}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=5, label="Customer review", placeholder="Paste a product review here..."),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    title="Amazon Review Sentiment",
    description=f"Classifies a product review as negative, neutral or positive. Model: {MODEL_NAME}.",
    examples=[
        ["Great tablet for the price. My kids love it and the battery lasts all day."],
        ["It works, but the screen is dimmer than I expected for the price."],
        ["Stopped charging after a week and support never replied. Avoid."],
    ],
)

if __name__ == "__main__":
    demo.launch()
