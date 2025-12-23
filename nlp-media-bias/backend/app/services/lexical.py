import json
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

BASE_STOPWORDS = set(stopwords.words("english"))

CUSTOM_STOPWORDS = {
    # journalisme
    "said", "via", "file", "files", "press", "service", "associated", "ap",
    "please", "refresh",

    # temps
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",

    # chiffres écrits
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"
}

STOPWORDS = BASE_STOPWORDS.union(CUSTOM_STOPWORDS)


def load_texts(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return [article["text"] for article in corpus]

def tokenize_corpus(texts: list) -> list:
    tokens = []
    for text in texts:
        words = word_tokenize(text)
        words = [
            w for w in words
            if w.isalpha() and w not in STOPWORDS and len(w) > 2
        ]
        tokens.extend(words)
    return tokens

def word_frequencies(tokens: list, top_n: int = 30) -> list:
    freq = Counter(tokens)
    return freq.most_common(top_n)

def ngram_frequencies(tokens: list, n: int = 2, top_n: int = 20) -> list:
    ng = ngrams(tokens, n)
    freq = Counter(ng)
    return freq.most_common(top_n)
