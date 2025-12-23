import json
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from app.services.lexical import STOPWORDS

def load_sentences(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    sentences = []

    for article in corpus:
        text = article["text"]
        for sent in sent_tokenize(text):
            tokens = [
                w for w in word_tokenize(sent.lower())
                if w.isalpha() and w not in STOPWORDS and len(w) > 2
            ]
            if len(tokens) >= 3:
                sentences.append(tokens)

    return sentences


def train_word2vec(sentences: list) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=3,
        workers=4,
        sg=1  # Skip-gram (meilleur pour petits corpus)
    )
    return model
