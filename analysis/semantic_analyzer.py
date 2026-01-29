# analysis/semantic_analyzer.py

import json
import os
from collections import defaultdict, Counter
import numpy as np
import yaml
import logging
import re
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, NMF, LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from scipy.sparse import vstack, csr_matrix
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from typing import List, Dict, Tuple, Set, Optional, Any

CUSTOM_STOPWORDS = set(ENGLISH_STOP_WORDS).union({
    "the", "to", "of", "in", "on", "for", "with", "as",
    "has", "have", "had", "is", "was", "were", "be",
    "from", "that", "this", "it", "its", "at", "by",
    "and", "or", "but", "not", "he", "she", "they", "we"
})

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedSemanticAnalyzer:
    """Analyseur sémantique avancé avec multiples techniques NLP"""

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.semantic_config = self.config["analysis"]["semantic"]
        
        # Configurations spécifiques
        self.w2v_config = self.semantic_config.get("word2vec", {
            "vector_size": 100,
            "window": 5,
            "min_count": 2,
            "workers": 4,
            "sg": 1
        })
        
        self.clustering_config = self.semantic_config.get("clustering", {
            "n_clusters": 4,
            "similarity_threshold": 0.85
        })
        
        self.lda_config = self.semantic_config.get("lda", {
            "n_topics": 5,
            "n_top_words": 10
        })

        self.corpus = []
        self.by_topic = defaultdict(list)
        self.topic_articles = defaultdict(list)

        # Vocabulaire étendu
        self.target_words = {
            "gaza": ["palestinians", "israelis", "civilians", "militants", "victims", 
                    "hamas", "israel", "gaza", "war", "conflict", "attack", "violence",
                    "children", "women", "refugees", "aid", "hospital", "school"],
            "ukraine": ["ukrainians", "russians", "civilians", "soldiers", "victims",
                       "putin", "zelensky", "ukraine", "russia", "war", "invasion",
                       "attack", "defense", "nato", "weapons", "sanctions", "refugees"]
        }
        
        # Catégories sémantiques pour analyse comparative
        self.semantic_categories = {
            "violence": ["attack", "violence", "bombing", "shelling", "strike", "raid"],
            "victims": ["civilian", "victim", "child", "woman", "family", "casualty"],
            "humanitarian": ["aid", "help", "assistance", "hospital", "school", "refugee"],
            "politics": ["government", "president", "minister", "diplomacy", "sanctions"],
            "military": ["army", "soldier", "troop", "weapon", "defense", "offensive"]
        }

        self.results = {
            "semantic_fields": {},
            "kwic": {},
            "word2vec": {},
            "semantic_clusters": {},
            "word_clusters": {},
            "topic_modeling": {},
            "actor_analysis": {},
            "comparative_analysis": {},
            "statistics": {}
        }

    def load_corpus(self, path):
        """Charge le corpus avec validation des données"""
        with open(path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        stats = defaultdict(int)
        for art in self.corpus:
            topic = art.get("topic", "unknown")
            content = art.get("content_clean", "")
            
            if content and len(content.split()) > 10:  # Filtre les textes trop courts
                self.by_topic[topic].append(content)
                self.topic_articles[topic].append(art)
                stats[topic] += 1

        logger.info(f"Corpus chargé: {len(self.corpus)} articles")
        logger.info(f"Distribution par topic: {dict(stats)}")
        
        # Statistiques de base
        total_words = sum(len(text.split()) for texts in self.by_topic.values() for text in texts)
        avg_words = total_words / len(self.corpus) if self.corpus else 0
        logger.info(f"Moyenne de mots par article: {avg_words:.1f}")

    def analyze_all(self, output_dir="analysis_results"):
        """Exécute toutes les analyses sémantiques"""
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Début de l'analyse sémantique complète")

        try:
            # Analyses fondamentales
            self.results["semantic_fields"] = self.enhanced_semantic_fields()
            self.results["kwic"] = self.enhanced_kwic()
            
            # Analyses vectorielles
            self.results["word2vec"] = self.enhanced_word2vec_analysis()
            
            # Clustering et topic modeling
            self.results["word_clusters"] = self.hierarchical_word_clustering()
            self.results["topic_modeling"] = self.topic_modeling_lda()
            self.results["semantic_clusters"] = self.hierarchical_semantic_clustering()
            
            # Analyses spécifiques
            self.results["actor_analysis"] = self.comprehensive_actor_analysis()
            self.results["comparative_analysis"] = self.cross_topic_comparative_analysis()
            
            # Calcul des statistiques
            self.results["statistics"] = self.calculate_statistics()

            # Sauvegarde
            timestamp = self._get_timestamp()
            path = os.path.join(output_dir, f"semantic_analysis_{timestamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f"Analyse sémantique sauvegardée: {path}")
            
            # Export CSV pour analyse externe
            self.export_to_csv(output_dir, timestamp)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse sémantique: {e}")
            raise

    def enhanced_semantic_fields(self, n_features=50):
        """Champs sémantiques avec TF-IDF et bigrammes"""
        output = {}
        for topic, texts in self.by_topic.items():
            if len(texts) < 3:
                logger.warning(f"Topic {topic}: trop peu de textes pour l'analyse TF-IDF")
                output[topic] = []
                continue
                
            # TF-IDF avec n-grammes
            vectorizer = TfidfVectorizer(
                stop_words=list(CUSTOM_STOPWORDS),
                max_features=n_features,
                min_df=2,
                max_df=0.95,
                ngram_range=(1, 2)  # Unigrams et bigrams
            )
            
            try:
                X = vectorizer.fit_transform(texts)
                features = vectorizer.get_feature_names_out()
                scores = np.asarray(X.mean(axis=0)).flatten()
                
                # Combiner features et scores
                features_with_scores = list(zip(features, scores))
                features_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Formater pour l'affichage
                output[topic] = [
                    {"word": word, "score": float(score)}
                    for word, score in features_with_scores[:n_features]
                ]
                
            except Exception as e:
                logger.error(f"Erreur TF-IDF pour {topic}: {e}")
                output[topic] = []

        return output

    def enhanced_kwic(self, window=8, max_examples=10):
        """Concordances avancées avec contexte étendu"""
        results = {}
        for topic, texts in self.by_topic.items():
            results[topic] = {}
            
            for word in self.target_words.get(topic, []):
                examples = []
                
                for text in texts[:100]:  # Limiter pour performance
                    sentences = re.split(r'[.!?]+', text)
                    
                    for sentence in sentences:
                        if word.lower() in sentence.lower():
                            words_in_sentence = sentence.split()
                            
                            # Trouver l'indice du mot
                            indices = [i for i, w in enumerate(words_in_sentence) 
                                      if w.lower() == word.lower()]
                            
                            for idx in indices:
                                left = words_in_sentence[max(0, idx - window):idx]
                                right = words_in_sentence[idx + 1:idx + window + 1]
                                full_sentence = ' '.join(words_in_sentence)
                                
                                examples.append({
                                    "sentence": full_sentence,
                                    "left": " ".join(left),
                                    "keyword": word,
                                    "right": " ".join(right),
                                    "position": idx,
                                    "sentence_length": len(words_in_sentence)
                                })
                                
                                if len(examples) >= max_examples * 2:
                                    break
                    
                    if len(examples) >= max_examples * 2:
                        break
                
                # Filtrer les meilleurs exemples
                if examples:
                    selected = examples[:max_examples]
                    
                    results[topic][word] = {
                        "examples": selected,
                        "total_occurrences": len(examples)
                    }
                else:
                    results[topic][word] = {
                        "examples": [],
                        "total_occurrences": 0
                    }
        
        return results

    def enhanced_word2vec_analysis(self):
        """Analyse Word2Vec avancée avec similarités sémantiques"""
        output = {}
        
        for topic, texts in self.by_topic.items():
            if len(texts) < 5:
                logger.warning(f"Topic {topic}: insuffisant pour Word2Vec")
                output[topic] = {}
                continue
            
            # Préparation des phrases
            raw_sentences = [text.lower().split() for text in texts]
            clean_sentences = self._clean_sentences(raw_sentences)
            
            if not clean_sentences:
                logger.warning(f"Topic {topic}: aucune phrase valide après nettoyage")
                output[topic] = {}
                continue
            
            # Entraînement du modèle
            model = Word2Vec(
                sentences=clean_sentences,
                vector_size=self.w2v_config["vector_size"],
                window=self.w2v_config["window"],
                min_count=self.w2v_config["min_count"],
                workers=self.w2v_config["workers"],
                sg=self.w2v_config["sg"],
                epochs=20,
                alpha=0.025,
                min_alpha=0.0001
            )
            
            topic_res = {}
            
            # Analyse pour chaque mot cible
            for word in self.target_words.get(topic, []):
                if word in model.wv:
                    # Similarités directes
                    similar_words = []
                    try:
                        similar_words = model.wv.most_similar(word, topn=10)
                    except:
                        pass
                    
                    # Distance sémantique entre paires intéressantes
                    semantic_distances = {}
                    comparison_pairs = [
                        ("civilians", "soldiers"),
                        ("victims", "attackers"),
                        ("children", "adults")
                    ]
                    
                    for w1, w2 in comparison_pairs:
                        if w1 in model.wv and w2 in model.wv:
                            similarity = model.wv.similarity(w1, w2)
                            semantic_distances[f"{w1}_vs_{w2}"] = float(similarity)
                    
                    topic_res[word] = {
                        "similar_words": [
                            {"word": w, "similarity": float(s)}
                            for w, s in similar_words
                        ],
                        "semantic_distances": semantic_distances,
                        "vector_norm": float(np.linalg.norm(model.wv[word]))
                    }
            
            output[topic] = topic_res
            
            # Analyse thématique additionnelle
            thematic_words = self._extract_thematic_words(model, topic)
            if thematic_words:
                output[topic]["_thematic_analysis"] = thematic_words
        
        return output

    def hierarchical_word_clustering(self, n_clusters=5):
        """Clustering hiérarchique des mots basé sur Word2Vec"""
        clusters = {}
        
        for topic, texts in self.by_topic.items():
            if len(texts) < 5:
                continue
            
            # Préparer les phrases et entraîner Word2Vec
            raw_sentences = [text.lower().split() for text in texts]
            clean_sentences = self._clean_sentences(raw_sentences)
            
            if not clean_sentences:
                continue
                
            model = Word2Vec(
                sentences=clean_sentences,
                vector_size=100,
                window=5,
                min_count=3,
                workers=4,
                sg=1
            )
            
            # Filtrer les mots pertinents
            words = [
                w for w in model.wv.index_to_key
                if w not in CUSTOM_STOPWORDS 
                and len(w) > 3 
                and any(c.isalpha() for c in w)
            ][:100]  # Limiter pour performance
            
            if len(words) < n_clusters:
                continue
            
            vectors = np.array([model.wv[w] for w in words])
            
            # Clustering hiérarchique
            Z = linkage(vectors, method='ward', metric='euclidean')
            labels = fcluster(Z, t=n_clusters, criterion='maxclust')
            
            # Organiser les clusters
            topic_clusters = {}
            for word, label in zip(words, labels):
                cluster_id = str(label)
                if cluster_id not in topic_clusters:
                    topic_clusters[cluster_id] = []
                topic_clusters[cluster_id].append(word)
            
            # Caractériser chaque cluster
            characterized_clusters = {}
            for cluster_id, cluster_words in topic_clusters.items():
                if cluster_words:
                    characterized_clusters[cluster_id] = {
                        "words": cluster_words[:15],
                        "size": len(cluster_words),
                        "coherence": self._calculate_cluster_coherence(cluster_words, model)
                    }
            
            clusters[topic] = characterized_clusters
        
        return clusters

    def topic_modeling_lda(self, n_topics=5, n_top_words=10):
        """Topic Modeling avec LDA"""
        topics = {}
        
        for topic, texts in self.by_topic.items():
            if len(texts) < 10:
                continue
            
            # Vectorisation
            vectorizer = CountVectorizer(
                max_df=0.95,
                min_df=2,
                stop_words=list(CUSTOM_STOPWORDS),
                max_features=1000
            )
            
            dtm = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42,
                n_jobs=-1
            )
            
            lda.fit(dtm)
            
            # Extraire les topics
            topic_results = {}
            for topic_idx, topic_weights in enumerate(lda.components_):
                top_indices = topic_weights.argsort()[:-n_top_words - 1:-1]
                top_words = [feature_names[i] for i in top_indices]
                top_weights = [float(topic_weights[i]) for i in top_indices]
                
                topic_results[str(topic_idx)] = {
                    "words": top_words,
                    "weights": top_weights,
                    "dominant_category": self._categorize_topic(top_words),
                    "coherence": np.mean(topic_weights[top_indices])
                }
            
            topics[topic] = topic_results
        
        return topics

    def hierarchical_semantic_clustering(self):
        """Clustering hiérarchique de phrases"""
        clusters = {}
        
        for topic, texts in self.by_topic.items():
            if len(texts) < 10:
                continue
            
            # Vectorisation TF-IDF
            vectorizer = TfidfVectorizer(
                stop_words=list(CUSTOM_STOPWORDS),
                max_features=500
            )
            
            X = vectorizer.fit_transform(texts)
            
            # Clustering hiérarchique - CONVERSION EN ARRAY NUMPY
            X_array = X.toarray()
            if X_array.shape[0] > 1:  # Besoin d'au moins 2 échantillons pour clustering
                Z = linkage(X_array, method='ward', metric='euclidean')
                
                # Découper en clusters
                max_clusters = min(5, len(texts) // 3)
                labels = fcluster(Z, t=max_clusters, criterion='maxclust')
                
                # Organiser les clusters
                topic_clusters = {}
                for cluster_id in set(labels):
                    indices = np.where(labels == cluster_id)[0]
                    cluster_texts = [texts[i] for i in indices]
                    
                    # Caractériser le cluster
                    if cluster_texts:
                        representative_texts = self._select_representative_texts(cluster_texts, X, indices)
                        
                        topic_clusters[str(cluster_id)] = {
                            "size": len(cluster_texts),
                            "representative_texts": representative_texts,
                            "all_texts_count": len(cluster_texts),
                            "dominant_words": self._extract_dominant_words(cluster_texts, vectorizer)
                        }
                
                clusters[topic] = topic_clusters
            else:
                clusters[topic] = {}
        
        return clusters

    def comprehensive_actor_analysis(self):
        """Analyse complète par acteur"""
        actors_analysis = {}
        
        for topic, texts in self.by_topic.items():
            if not texts:
                continue
            
            actors_analysis[topic] = {}
            
            for actor in self.target_words.get(topic, []):
                # Extraire les phrases contenant l'acteur
                actor_sentences = []
                for text in texts:
                    sentences = re.split(r'[.!?]+', text)
                    for sentence in sentences:
                        if actor.lower() in sentence.lower():
                            actor_sentences.append(sentence.strip())
                
                if len(actor_sentences) < 3:
                    continue
                
                # TF-IDF spécifique à l'acteur
                actor_vectorizer = TfidfVectorizer(
                    stop_words=list(CUSTOM_STOPWORDS),
                    max_features=20
                )
                actor_tfidf = actor_vectorizer.fit_transform(actor_sentences)
                actor_features = actor_vectorizer.get_feature_names_out()
                actor_scores = np.asarray(actor_tfidf.mean(axis=0)).flatten()
                
                # Mots associés (triés par score TF-IDF)
                associated_words = [
                    {"word": word, "score": float(score)}
                    for word, score in zip(actor_features, actor_scores)
                    if word != actor
                ]
                associated_words.sort(key=lambda x: x["score"], reverse=True)
                
                # Concordances
                concordances = []
                for sentence in actor_sentences[:10]:  # Limiter
                    words = sentence.split()
                    for i, w in enumerate(words):
                        if w.lower() == actor.lower():
                            left = ' '.join(words[max(0, i-5):i])
                            right = ' '.join(words[i+1:i+6])
                            concordances.append({
                                "sentence": sentence[:200],  # Tronquer
                                "left": left,
                                "keyword": actor,
                                "right": right
                            })
                            break
                
                # Co-occurrences fréquentes
                cooccurrences = Counter()
                for sentence in actor_sentences:
                    words = sentence.lower().split()
                    actor_index = next((i for i, w in enumerate(words) if w == actor), -1)
                    if actor_index >= 0:
                        context_words = words[max(0, actor_index-5):actor_index] + \
                                      words[actor_index+1:actor_index+6]
                        cooccurrences.update([
                            w for w in context_words 
                            if w not in CUSTOM_STOPWORDS and w != actor and len(w) > 2
                        ])
                
                actors_analysis[topic][actor] = {
                    "total_mentions": len(actor_sentences),
                    "associated_words": associated_words[:15],
                    "concordances": concordances[:5],
                    "top_cooccurrences": dict(cooccurrences.most_common(10)),
                    "average_sentence_length": np.mean([len(s.split()) for s in actor_sentences]) if actor_sentences else 0
                }
        
        return actors_analysis

    def cross_topic_comparative_analysis(self):
        """Analyse comparative entre les topics"""
        comparison = {
            "semantic_overlap": {},
            "word_usage_differences": {},
            "thematic_focus": {},
            "similarity_scores": {}
        }
        
        topics = list(self.by_topic.keys())
        if len(topics) < 2:
            return comparison
        
        # 1. Chevauchement sémantique
        for topic in topics:
            if topic in self.results.get("semantic_fields", {}):
                words = [item["word"] if isinstance(item, dict) else item 
                        for item in self.results["semantic_fields"][topic][:20]]
                comparison["semantic_overlap"][topic] = words
        
        # 2. Différences d'usage pour les mots communs
        common_actors = set(self.target_words.get("gaza", [])).intersection(
                        set(self.target_words.get("ukraine", [])))
        
        for actor in common_actors:
            usage_diff = {}
            for topic in topics:
                if (topic in self.results.get("actor_analysis", {}) and 
                    actor in self.results["actor_analysis"][topic]):
                    actor_data = self.results["actor_analysis"][topic][actor]
                    usage_diff[topic] = {
                        "mentions": actor_data.get("total_mentions", 0),
                        "avg_context_length": actor_data.get("average_sentence_length", 0)
                    }
            
            if usage_diff:
                comparison["word_usage_differences"][actor] = usage_diff
        
        # 3. Focus thématique
        for topic in topics:
            thematic_focus = {}
            for category, terms in self.semantic_categories.items():
                category_score = 0
                if topic in self.results.get("semantic_fields", {}):
                    topic_words = [item["word"] if isinstance(item, dict) else item 
                                 for item in self.results["semantic_fields"][topic]]
                    for term in terms:
                        if term in topic_words:
                            category_score += 1
                thematic_focus[category] = category_score
            
            comparison["thematic_focus"][topic] = thematic_focus
        
        # 4. Scores de similarité globale
        if len(topics) >= 2:
            # Similarité cosinus entre les vecteurs TF-IDF agrégés
            all_texts = []
            topic_labels = []
            for topic in topics:
                all_texts.extend(self.by_topic[topic])
                topic_labels.extend([topic] * len(self.by_topic[topic]))
            
            if len(all_texts) > 0:
                vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
                X = vectorizer.fit_transform(all_texts)
                
                # Moyenne par topic
                topic_vectors = {}
                for topic in topics:
                    indices = [i for i, label in enumerate(topic_labels) if label == topic]
                    if indices:
                        # Calculer la moyenne et convertir en array numpy
                        topic_matrix = X[indices]
                        if hasattr(topic_matrix, "toarray"):
                            topic_mean = topic_matrix.mean(axis=0)
                            # Convertir en array numpy 2D
                            topic_vectors[topic] = np.asarray(topic_mean).reshape(1, -1)
                        else:
                            # Déjà un array numpy
                            topic_vectors[topic] = topic_matrix.mean(axis=0).reshape(1, -1)
                
                # Calculer les similarités
                similarity_matrix = {}
                for t1 in topics:
                    for t2 in topics:
                        if t1 in topic_vectors and t2 in topic_vectors:
                            # S'assurer que ce sont des arrays numpy 2D
                            vec1 = np.asarray(topic_vectors[t1]).reshape(1, -1)
                            vec2 = np.asarray(topic_vectors[t2]).reshape(1, -1)
                            
                            # Calculer la similarité cosinus
                            sim = cosine_similarity(vec1, vec2)[0][0]
                            similarity_matrix[f"{t1}_vs_{t2}"] = float(sim)
                
                comparison["similarity_scores"] = similarity_matrix
        
        return comparison

    def calculate_statistics(self):
        """Calcule les statistiques descriptives"""
        stats = {
            "corpus": {
                "total_articles": len(self.corpus),
                "articles_by_topic": {topic: len(articles) 
                                    for topic, articles in self.by_topic.items()},
                "avg_words_per_article": {},
                "vocabulary_size": {},
                "lexical_diversity": {}
            },
            "semantic_metrics": {}
        }
        
        for topic, texts in self.by_topic.items():
            if texts:
                # Statistiques de base
                total_words = sum(len(text.split()) for text in texts)
                avg_words = total_words / len(texts)
                
                # Taille du vocabulaire
                all_words = [word for text in texts for word in text.lower().split()]
                vocab_size = len(set(all_words))
                
                # Diversité lexicale (Type-Token Ratio)
                ttr = vocab_size / len(all_words) if all_words else 0
                
                stats["corpus"]["avg_words_per_article"][topic] = float(avg_words)
                stats["corpus"]["vocabulary_size"][topic] = vocab_size
                stats["corpus"]["lexical_diversity"][topic] = float(ttr)
                
                # Métriques sémantiques
                if topic in self.results.get("semantic_fields", {}):
                    semantic_words = len(self.results["semantic_fields"][topic])
                    if semantic_words > 0:
                        scores = []
                        for item in self.results["semantic_fields"][topic]:
                            if isinstance(item, dict) and "score" in item:
                                scores.append(item["score"])
                        avg_score = np.mean(scores) if scores else 0
                        stats["semantic_metrics"][topic] = {
                            "distinct_semantic_fields": semantic_words,
                            "avg_tfidf_score": float(avg_score)
                        }
        
        return stats

    def export_to_csv(self, output_dir, timestamp):
        """Exporte les résultats principaux en CSV"""
        try:
            import pandas as pd
            
            # 1. Champs sémantiques
            semantic_data = []
            for topic, items in self.results.get("semantic_fields", {}).items():
                for item in items:
                    if isinstance(item, dict):
                        semantic_data.append({
                            "topic": topic,
                            "word": item.get("word", ""),
                            "score": item.get("score", 0)
                        })
            
            if semantic_data:
                df_semantic = pd.DataFrame(semantic_data)
                df_semantic.to_csv(
                    os.path.join(output_dir, f"semantic_fields_{timestamp}.csv"),
                    index=False, encoding='utf-8'
                )
            
            # 2. Analyse par acteur
            actor_data = []
            for topic, actors in self.results.get("actor_analysis", {}).items():
                for actor, data in actors.items():
                    actor_data.append({
                        "topic": topic,
                        "actor": actor,
                        "mentions": data.get("total_mentions", 0),
                        "avg_context_length": data.get("average_sentence_length", 0)
                    })
            
            if actor_data:
                df_actors = pd.DataFrame(actor_data)
                df_actors.to_csv(
                    os.path.join(output_dir, f"actor_analysis_{timestamp}.csv"),
                    index=False, encoding='utf-8'
                )
            
            logger.info("Export CSV terminé")
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'export CSV: {e}")

    # ========== MÉTHODES AUXILIAIRES ==========
    
    def _clean_sentences(self, sentences):
        """Nettoie les phrases pour Word2Vec"""
        cleaned = []
        for sent in sentences:
            words = []
            for word in sent:
                # Nettoyage basique
                clean_word = re.sub(r'[^a-zA-Z]', '', word.lower())
                if (clean_word and 
                    len(clean_word) > 1 and 
                    clean_word not in CUSTOM_STOPWORDS):
                    words.append(clean_word)
            if words:
                cleaned.append(words)
        return cleaned
    
    def _extract_thematic_words(self, model, topic):
        """Extrait des mots thématiques basés sur les catégories sémantiques"""
        thematic_words = {}
        
        for category, terms in self.semantic_categories.items():
            category_words = []
            for term in terms:
                if term in model.wv:
                    try:
                        similar = model.wv.most_similar(term, topn=5)
                        category_words.extend([w for w, _ in similar])
                    except:
                        pass
            
            # Fréquence et unicité
            if category_words:
                freq = Counter(category_words)
                thematic_words[category] = [
                    {"word": word, "frequency": count}
                    for word, count in freq.most_common(5)
                ]
        
        return thematic_words
    
    def _calculate_cluster_coherence(self, words, model):
        """Calcule la cohérence d'un cluster de mots"""
        if len(words) < 2:
            return 0
        
        similarities = []
        for i, w1 in enumerate(words):
            if w1 in model.wv:
                for w2 in words[i+1:]:
                    if w2 in model.wv:
                        try:
                            sim = model.wv.similarity(w1, w2)
                            similarities.append(sim)
                        except:
                            pass
        
        return float(np.mean(similarities)) if similarities else 0
    
    def _categorize_topic(self, words):
        """Catégorise un topic basé sur ses mots-clés"""
        category_scores = {}
        
        for category, terms in self.semantic_categories.items():
            score = sum(1 for term in terms if term in words)
            category_scores[category] = score
        
        if category_scores:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return "unknown"
    
    def _select_representative_texts(self, texts, tfidf_matrix, indices, n=3):
        """Sélectionne les textes les plus représentatifs d'un cluster"""
        if not texts or len(texts) < n:
            return texts
        
        # Calculer la similarité moyenne avec les autres textes du cluster
        cluster_matrix = tfidf_matrix[indices].toarray()  # Conversion en array numpy
        similarities = cosine_similarity(cluster_matrix)
        avg_similarities = similarities.mean(axis=1)
        
        # Sélectionner les textes avec la plus haute similarité moyenne
        top_indices = avg_similarities.argsort()[-n:][::-1]
        
        return [texts[i][:300] for i in top_indices]  # Tronquer
    
    def _extract_dominant_words(self, texts, vectorizer, n=10):
        """Extrait les mots dominants d'un ensemble de textes"""
        combined_text = ' '.join(texts)
        tfidf = vectorizer.transform([combined_text])
        feature_names = vectorizer.get_feature_names_out()
        
        scores = tfidf.toarray().flatten()
        top_indices = scores.argsort()[-n:][::-1]
        
        return [
            {"word": feature_names[i], "score": float(scores[i])}
            for i in top_indices if scores[i] > 0
        ]
    
    def _get_timestamp(self):
        """Génère un timestamp pour les noms de fichiers"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    """Fonction principale de test"""
    import glob
    
    files = glob.glob("preprocessed/preprocessed_*.json")
    if not files:
        print("Aucun corpus trouvé. Exécutez d'abord le prétraitement.")
        return
    
    # Sélectionner le fichier le plus récent
    latest_file = max(files)
    print(f"Analyse du corpus: {latest_file}")
    
    # Initialiser et exécuter l'analyse
    analyzer = EnhancedSemanticAnalyzer()
    analyzer.load_corpus(latest_file)
    
    try:
        results = analyzer.analyze_all()
        
        # Afficher un résumé
        print("\n" + "="*60)
        print("RÉSUMÉ DE L'ANALYSE SÉMANTIQUE")
        print("="*60)
        
        for topic, stats in results.get("statistics", {}).get("corpus", {}).get("articles_by_topic", {}).items():
            print(f"\n{topic.upper()}:")
            print(f"  Articles: {stats}")
            print(f"  Diversité lexicale: {results['statistics']['corpus']['lexical_diversity'].get(topic, 0):.3f}")
            
            if topic in results.get("actor_analysis", {}):
                actors = list(results["actor_analysis"][topic].keys())[:3]
                print(f"  Acteurs principaux: {', '.join(actors)}")
        
        # Similarité entre topics
        if "comparative_analysis" in results:
            sim_scores = results["comparative_analysis"].get("similarity_scores", {})
            for key, score in sim_scores.items():
                if "_vs_" in key:
                    print(f"  Similarité {key}: {score:.3f}")
        
    except Exception as e:
        print(f"Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()