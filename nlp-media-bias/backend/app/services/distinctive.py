import json
import math
from collections import Counter
from app.services.lexical import tokenize_corpus, load_texts

def compute_frequencies(file_path: str) -> Counter:
    texts = load_texts(file_path)
    tokens = tokenize_corpus(texts)
    return Counter(tokens)

def log_ratio(freq_a: Counter, freq_b: Counter, top_n: int = 30):
    scores = {}

    vocab = set(freq_a.keys()).union(set(freq_b.keys()))

    for word in vocab:
        score = math.log((freq_a[word] + 1) / (freq_b[word] + 1))
        scores[word] = score

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_a = sorted_scores[:top_n]
    top_b = sorted_scores[-top_n:]

    return top_a, top_b
