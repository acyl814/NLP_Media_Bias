from app.services.semantic import load_sentences, train_word2vec
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))


KEYWORDS = [
    "soldier",
    "soldiers",
    "militant",
    "militants",
    "civilian",
    "civilians",
    "child",
    "children",
    "victim",
    "victims",
    "attack"
]

def analyze(name: str, json_file: str):
    print(f"\n===== {name.upper()} SEMANTIC ANALYSIS =====")

    sentences = load_sentences(json_file)
    model = train_word2vec(sentences)

    for word in KEYWORDS:
        if word in model.wv:
            print(f"\nSimilar words to '{word}':")
            for sim_word, score in model.wv.most_similar(word, topn=5):
                print(f"  {sim_word:15} {score:.3f}")
        else:
            print(f"\n'{word}' not in vocabulary.")


if __name__ == "__main__":
    analyze("Ukraine", "data/processed/ukraine_clean.json")
    analyze("Gaza", "data/processed/gaza_clean.json")
import json
from pathlib import Path
from gensim.models import Word2Vec

# =========================
# Load cleaned corpora
# =========================
def load_texts(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = []
    for doc in data:
        tokens = [
            w.lower()
            for w in doc["text"].split()
            if w.isalpha() and w.lower() not in STOPWORDS
        ]
        texts.append(tokens)

    return texts


ukraine_texts = load_texts("data/processed/ukraine_clean.json")
gaza_texts = load_texts("data/processed/gaza_clean.json")

# =========================
# Train Word2Vec models
# =========================
model_ukraine = Word2Vec(
    sentences=ukraine_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

model_gaza = Word2Vec(
    sentences=gaza_texts,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4
)

# =========================
# Safe similarity function
# =========================
def safe_similar(model, word, topn=5):
    if word in model.wv:
        return model.wv.most_similar(word, topn=topn)
    return []

# =========================
# Extract semantic relations
# =========================
semantic_results = {
    "ukraine": {
        "soldiers": safe_similar(model_ukraine, "soldiers"),
        "civilians": safe_similar(model_ukraine, "civilians"),
        "attack": safe_similar(model_ukraine, "attack"),
    },
    "gaza": {
        "militants": safe_similar(model_gaza, "militants"),
        "children": safe_similar(model_gaza, "children"),
        "attack": safe_similar(model_gaza, "attack"),
    }
}

# =========================
# Convert to JSON-safe format
# =========================
for corpus in semantic_results:
    for concept in semantic_results[corpus]:
        semantic_results[corpus][concept] = [
            {"term": w, "score": float(s)}
            for w, s in semantic_results[corpus][concept]
        ]

# =========================
# Save results
# =========================
output_path = Path("data/results/semantic_similarities.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(semantic_results, f, indent=2)

print(f"Semantic analysis saved to {output_path}")
