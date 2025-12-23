import json
from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def analyze_sentiment(text: str) -> dict:
    return sia.polarity_scores(text)

def analyze_corpus(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    results = []

    for article in corpus:
        scores = analyze_sentiment(article["text"])
        results.append(scores)

    return results

def average_sentiment(scores: list) -> dict:
    avg = {
        "neg": 0.0,
        "neu": 0.0,
        "pos": 0.0,
        "compound": 0.0
    }

    for s in scores:
        for key in avg:
            avg[key] += s[key]

    for key in avg:
        avg[key] /= len(scores)

    return avg
