import json
from pathlib import Path
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# =========================
# Setup
# =========================
nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# =========================
# Load cleaned corpora
# =========================
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [doc["text"] for doc in data]

ukraine_texts = load_texts("data/processed/ukraine_clean.json")
gaza_texts = load_texts("data/processed/gaza_clean.json")

# =========================
# Compute sentiment scores
# =========================
def compute_average_sentiment(texts):
    scores = {"neg": 0, "neu": 0, "pos": 0, "compound": 0}
    for text in texts:
        s = sia.polarity_scores(text)
        for k in scores:
            scores[k] += s[k]

    n = len(texts)
    return {k: scores[k] / n for k in scores}

ukraine_scores = compute_average_sentiment(ukraine_texts)
gaza_scores = compute_average_sentiment(gaza_texts)

# =========================
# Save results
# =========================
sentiment_results = {
    "ukraine": ukraine_scores,
    "gaza": gaza_scores
}

output_path = Path("data/results/sentiment_scores.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sentiment_results, f, indent=2)

print(f"Sentiment results saved to {output_path}")
