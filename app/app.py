"""Gradio demo for the sentiment classifier (Task 4).

Runs locally with `python app/app.py` and unchanged on Hugging Face Spaces.
Point MODEL_ID at your fine-tuned model: a Hub repo id on Spaces, or a local
path such as `../models/sentiment-distilbert` when running from this folder.
"""

import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "distilbert-base-uncased-finetuned-sst-2-english")

classifier = pipeline("text-classification", model=MODEL_ID, top_k=None)

# Raw label ids (LABEL_0, ...) are unhelpful in a demo; map them to words.
PRETTY = {
    "LABEL_0": "negative",
    "LABEL_1": "neutral",
    "LABEL_2": "positive",
}


def predict(review: str) -> dict[str, float]:
    review = (review or "").strip()
    if not review:
        return {}
    scores = classifier(review)[0]
    return {PRETTY.get(s["label"], s["label"].lower()): float(s["score"]) for s in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=6,
        label="Customer review",
        placeholder="Paste a product review here...",
    ),
    outputs=gr.Label(num_top_classes=3, label="Predicted sentiment"),
    title="Amazon Review Sentiment",
    description="Classifies a product review as negative, neutral, or positive.",
    examples=[
        ["Battery lasts for days and setup took two minutes. Worth every cent."],
        ["It works, but the screen is dimmer than I expected for the price."],
        ["Stopped charging after a week and support never replied. Avoid."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
