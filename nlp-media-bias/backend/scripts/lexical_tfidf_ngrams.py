import json
from app.services.lexical_tfidf_ngrams import compute_tfidf_ngrams

def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for doc in data:
        if "clean_text" in doc:
            texts.append(doc["clean_text"])
        elif "cleaned_text" in doc:
            texts.append(doc["cleaned_text"])
        elif "text" in doc:
            texts.append(doc["text"])
        else:
            raise KeyError("No text field found in document")

    return texts

ukraine_texts = load_texts("data/processed/ukraine_clean.json")
gaza_texts = load_texts("data/processed/gaza_clean.json")

print("===== TF-IDF N-GRAMS — UKRAINE =====")
for term, score in compute_tfidf_ngrams(ukraine_texts):
    print(f"{term:<30} {score:.4f}")

print("\n===== TF-IDF N-GRAMS — GAZA =====")
for term, score in compute_tfidf_ngrams(gaza_texts):
    print(f"{term:<30} {score:.4f}")


import json
from pathlib import Path

output = {
    "ukraine": [
        {"term": term, "score": float(score)}
        for term, score in compute_tfidf_ngrams(ukraine_texts)
    ],
    "gaza": [
        {"term": term, "score": float(score)}
        for term, score in compute_tfidf_ngrams(gaza_texts)
    ]
}

output_path = Path("data/results/tfidf_ngrams.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print(f"\nTF-IDF results saved to {output_path}")
