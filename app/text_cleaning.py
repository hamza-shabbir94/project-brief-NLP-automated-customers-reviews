"""Text cleaning for the TF-IDF sentiment model (Task 1 / Task 4).

v2 changes (see notebook 02, "Improving the model"):
  * negation words are KEPT ("not", "no", "too", "very", "never", ...):
    the default stopword list removed them, so "not good" became "good";
  * "n't" is expanded to " not" ("didn't" -> "did not");
  * no lemmatizing (it did not help macro-F1).

Both notebook 02 (training) and app.py (serving) import this file, so the
saved model cleans text exactly the same way in both places.
This file must sit next to app.py on the Hugging Face Space.
"""
import re

import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# Words that flip or strengthen sentiment: keep them
NEGATIONS = {
    "no", "not", "nor", "never", "too", "very", "but", "against", "only", "few",
    "more", "most", "off", "down", "up", "out", "over", "again",
    # pieces left over from "don't", "didn't", ... (kept as negation signals)
    "don", "didn", "doesn", "isn", "wasn", "won", "couldn", "wouldn", "shouldn",
    "aren", "haven", "hasn", "weren", "mustn", "needn", "ain",
}
STOP_WORDS = set(stopwords.words("english")) - NEGATIONS


def remove_html_tags(text):
    text = re.sub(r"<script.*?</script>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    return re.sub(r"<[^>]+>", " ", text)


def clean_one(text):
    """Full cleaning for one review."""
    text = remove_html_tags(str(text)).lower()
    text = text.replace("’", "'").replace("n't", " not")   # curly apostrophe, then didn't -> did not
    text = re.sub(r"[^a-z\s]", " ", text)      # letters only
    text = re.sub(r"\b[a-z]\b", " ", text)     # single letters
    return " ".join(w for w in text.split() if w not in STOP_WORDS)


def preprocess(texts):
    """Clean a list / Series of reviews. First step of the saved sklearn Pipeline."""
    return [clean_one(t) for t in texts]
