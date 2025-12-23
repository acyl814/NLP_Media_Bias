import json
import math
from collections import Counter
from pathlib import Path

# =========================
# Load cleaned corpora
# =========================
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return " ".join(doc["text"].lower() for doc in data).split()

ukraine_tokens = load_texts("data/processed/ukraine_clean.json")
gaza_tokens = load_texts("data/processed/gaza_clean.json")

# =========================
# Compute frequencies
# =========================
ukraine_freq = Counter(ukraine_tokens)
gaza_freq = Counter(gaza_tokens)

# =========================
# Compute log-ratio
# =========================
def log_ratio(word, freq_a, freq_b, alpha=0.01):
    a = freq_a.get(word, 0) + alpha
    b = freq_b.get(word, 0) + alpha
    return math.log2(a / b)

all_words = set(ukraine_freq) | set(gaza_freq)

scores = []
for word in all_words:
    score = log_ratio(word, ukraine_freq, gaza_freq)
    scores.append((word, score))

# =========================
# Sort & select
# =========================
scores_sorted = sorted(scores, key=lambda x: x[1], reverse=True)

top_ukraine_words = scores_sorted[:30]
top_gaza_words = scores_sorted[-30:]

# =========================
# Save results
# =========================
distinctive_results = {
    "ukraine": [
        {"term": w, "score": float(s)}
        for w, s in top_ukraine_words
    ],
    "gaza": [
        {"term": w, "score": float(s)}
        for w, s in reversed(top_gaza_words)
    ]
}

output_path = Path("data/results/distinctive_words.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(distinctive_results, f, indent=2)

print(f"Distinctive words saved to {output_path}")
