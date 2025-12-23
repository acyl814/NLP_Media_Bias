from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def compute_tfidf_ngrams(texts, top_k=20):
    """
    Compute TF-IDF scores for unigrams + bigrams
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X = vectorizer.fit_transform(texts)
    scores = np.asarray(X.mean(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    ranked = sorted(
        zip(terms, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_k]
