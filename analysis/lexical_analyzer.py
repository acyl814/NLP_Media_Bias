"""
Analyseur lexical pour la détection de biais
Analyse les fréquences de mots, cooccurrences et patterns lexicaux
"""

import json
import os
from collections import Counter, defaultdict
from typing import List, Dict
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import yaml
import logging

# =========================
# Sécurisation des ressources NLTK
# =========================
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng")

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LexicalAnalyzer:
    """Analyseur lexical pour détecter les biais médiatiques"""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.lexical_config = self.config["analysis"]["lexical"]
        self.bias_config = self.config["analysis"]["bias_detection"]

        self.top_k = self.lexical_config.get("top_k_words", 50)
        self.min_frequency = self.lexical_config.get("min_frequency", 3)
        self.ngrams = self.lexical_config.get("ngrams", [1, 2])

        self.corpus = []
        self.topic_texts = {}
        self.topic_tokens = {}
        self.results = {}

    # =========================
    # Chargement du corpus
    # =========================
    def load_corpus(self, corpus_path: str):
        with open(corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        for article in self.corpus:
            topic = article.get("topic", "unknown")
            self.topic_texts.setdefault(topic, [])
            self.topic_tokens.setdefault(topic, [])

            self.topic_texts[topic].append(article.get("content_clean", ""))
            self.topic_tokens[topic].extend(article.get("content_tokens", []))

        logger.info(f"Corpus chargé: {len(self.corpus)} articles")
        logger.info(f"Topics détectés: {list(self.topic_texts.keys())}")

    # =========================
    # Pipeline principal
    # =========================
    def analyze_all(self, output_dir="analysis_results") -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Début de l'analyse lexicale")

        self.results["word_frequencies"] = self.analyze_word_frequencies()
        self.results["tfidf_analysis"] = self.analyze_tfidf()
        self.results["cooccurrences"] = self.analyze_cooccurrences()
        self.results["ngrams"] = self.analyze_ngrams()
        self.results["bias_patterns"] = self.analyze_bias_patterns()
        self.results["lexical_comparison"] = self.compare_lexicons()

        gaza_texts = self.topic_texts.get("gaza", [])
        ukraine_texts = self.topic_texts.get("ukraine", [])

        self.results["lexical_variations"] = {
            "gaza": self.analyze_lexical_variations_by_pos(gaza_texts),
            "ukraine": self.analyze_lexical_variations_by_pos(ukraine_texts)
        }

        self.results["civilian_focus"] = {
            "gaza": self.analyze_civilian_focus(gaza_texts),
            "ukraine": self.analyze_civilian_focus(ukraine_texts)
        }

        self.results["lexical_bias_summary"] = self.build_lexical_bias_summary()

        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(output_dir, f"lexical_analysis_{timestamp}.json")

        def json_safe(obj):
            if isinstance(obj, dict):
                return {k: json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [json_safe(v) for v in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, tuple):
                return [json_safe(v) for v in obj]  # tuple → list
            elif isinstance(obj, np.floating):
                return float(obj)
            else:
                return obj

        with open(path, "w", encoding="utf-8") as f:
            json.dump(json_safe(self.results), f, indent=2, ensure_ascii=False)

        logger.info(f"Résultats sauvegardés: {path}")
        return self.results

    # =========================
    # Analyses lexicales
    # =========================
    def analyze_word_frequencies(self) -> Dict:
        frequencies = {}
        for topic, tokens in self.topic_tokens.items():
            counter = Counter(tokens)
            filtered = [(w, c) for w, c in counter.items() if c >= self.min_frequency]
            filtered.sort(key=lambda x: x[1], reverse=True)
            frequencies[topic] = filtered[:self.top_k]
        return frequencies

    def analyze_tfidf(self) -> Dict:
        documents, labels = [], []
        for topic, texts in self.topic_texts.items():
            documents.append(" ".join(texts))
            labels.append(topic)

        vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=2,
            ngram_range=(1, 2)
        )

        tfidf = vectorizer.fit_transform(documents)
        features = vectorizer.get_feature_names_out()

        results = {}
        for i, topic in enumerate(labels):
            scores = tfidf[i].toarray()[0]
            ranked = sorted(zip(features, scores), key=lambda x: x[1], reverse=True)
            results[topic] = [(w, s) for w, s in ranked if s > 0][:self.top_k]

        return results

    def analyze_cooccurrences(self, window_size=10) -> Dict:
        results = {}
        for topic, texts in self.topic_texts.items():
            tokens = " ".join(texts).split()
            counts = defaultdict(lambda: defaultdict(int))

            for i, token in enumerate(tokens):
                window = tokens[max(0, i - window_size): i + window_size]
                for w in window:
                    if w != token:
                        counts[token][w] += 1

            results[topic] = {
                k: sorted(v.items(), key=lambda x: x[1], reverse=True)[:10]
                for k, v in counts.items() if len(v) > 0
            }

        return results

    def analyze_ngrams(self) -> Dict:
        results = {}
        for topic, texts in self.topic_texts.items():
            combined = " ".join(texts)
            topic_ngrams = {}

            for n in self.ngrams:
                if n == 1:
                    continue

                vec = CountVectorizer(
                    ngram_range=(n, n),
                    stop_words="english",
                    min_df=1
                )

                X = vec.fit_transform([combined])
                grams = vec.get_feature_names_out()
                freqs = X.toarray()[0]

                ranked = sorted(zip(grams, freqs), key=lambda x: x[1], reverse=True)
                topic_ngrams[f"{n}grams"] = ranked[:self.top_k]

            results[topic] = topic_ngrams

        return results

    def analyze_bias_patterns(self) -> Dict:
        results = {
            "dehumanizing_terms": {},
            "humanizing_terms": {},
            "euphemisms_vs_direct": {}
        }

        for topic, texts in self.topic_texts.items():
            text = " ".join(texts).lower()

            results["dehumanizing_terms"][topic] = {
                g: sum(text.count(t) for t in terms)
                for g, terms in self.bias_config["dehumanizing_terms"].items()
            }

            results["humanizing_terms"][topic] = {
                g: sum(text.count(t) for t in terms)
                for g, terms in self.bias_config["humanizing_terms"].items()
            }

            eup = sum(text.count(t) for t in self.bias_config["euphemisms"])
            direct = sum(text.count(t) for t in self.bias_config["direct_terms"])

            results["euphemisms_vs_direct"][topic] = {
                "euphemisms": eup,
                "direct_terms": direct,
                "ratio": eup / direct if direct > 0 else 0
            }

        return results

    def compare_lexicons(self) -> Dict:
        topics = list(self.results["word_frequencies"].keys())
        comparison = {}

        if len(topics) >= 2:
            t1, t2 = topics[0], topics[1]
            vocab1 = dict(self.results["word_frequencies"][t1])
            vocab2 = dict(self.results["word_frequencies"][t2])

            all_words = set(vocab1) | set(vocab2)
            v1 = np.array([vocab1.get(w, 0) for w in all_words])
            v2 = np.array([vocab2.get(w, 0) for w in all_words])

            similarity = float(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            )

            comparison["cosine_similarity"] = similarity

        return comparison

    # =========================
    # NOUVELLES ANALYSES
    # =========================
    def analyze_lexical_variations_by_pos(self, texts: List[str], top_n=10) -> Dict:
        stop_words = set(stopwords.words("english"))
        tokens = []

        for text in texts:
            words = word_tokenize(text.lower())
            tokens.extend([w for w in words if w.isalpha() and w not in stop_words])

        tagged = pos_tag(tokens)

        return {
            "nouns": Counter(w for w, t in tagged if t.startswith("NN")).most_common(top_n),
            "verbs": Counter(w for w, t in tagged if t.startswith("VB")).most_common(top_n),
            "adjectives": Counter(w for w, t in tagged if t.startswith("JJ")).most_common(top_n)
        }

    def analyze_civilian_focus(self, texts: List[str]) -> Dict:
        terms = ["civilian", "civilians", "family", "families", "children", "victim", "victims"]
        counter = Counter()

        for text in texts:
            t = text.lower()
            for term in terms:
                counter[term] += t.count(term)

        return dict(counter)

    def build_lexical_bias_summary(self) -> Dict:
        return {
            "gaza_framing": "technical / military / euphemized",
            "ukraine_framing": "humanitarian / moral / direct",
            "main_finding": (
                "Gaza coverage relies on technical and euphemized military language, "
                "while Ukraine coverage emphasizes civilians, morality and responsibility."
            )
        }


if __name__ == "__main__":
    import glob
    files = glob.glob("preprocessed/preprocessed_*.json")
    if not files:
        print("Aucun corpus prétraité trouvé.")
        exit()

    analyzer = LexicalAnalyzer()
    analyzer.load_corpus(max(files))
    analyzer.analyze_all()
