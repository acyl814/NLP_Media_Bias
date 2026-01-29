#!/usr/bin/env python3
"""
Script de régénération COMPLÈTE de l'analyse sémantique - VERSION CORRIGÉE
"""

import sys
import os
from pathlib import Path

# Déterminer le chemin absolu du projet
current_file = Path(__file__).resolve()
project_root = current_file.parent  # Racine de nlp_bias_analysis
parent_root = project_root.parent   # Répertoire parent

# Ajouter les chemins nécessaires
sys.path.insert(0, str(parent_root))  # Pour 'OKComputer_NLP HPC Project Implementation'
sys.path.insert(0, str(project_root)) # Pour 'nlp_bias_analysis'

print(f"🔧 Configuration des chemins:")
print(f"   Fichier courant: {current_file}")
print(f"   Racine projet: {project_root}")
print(f"   Racine parent: {parent_root}")
print(f"   PYTHONPATH: {sys.path[:3]}")

import json
import yaml
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Essayer d'importer avec plusieurs méthodes
try:
    # Méthode 1: Import direct
    from corpus_loader import CorpusLoader
    from text_preprocessor import TextPreprocessor
    logger.info("✅ Imports réussis (méthode directe)")
except ImportError:
    try:
        # Méthode 2: Import depuis le package
        from nlp_bias_analysis.corpus_loader import CorpusLoader
        from nlp_bias_analysis.text_preprocessor import TextPreprocessor
        logger.info("✅ Imports réussis (méthode package)")
    except ImportError as e:
        logger.error(f"❌ Échec des imports: {e}")
        logger.error("Vérifiez la structure du projet:")
        logger.error(f"1. Répertoire courant: {os.getcwd()}")
        logger.error(f"2. Contenu de {project_root}:")
        for f in project_root.iterdir():
            logger.error(f"   - {f.name}")
        sys.exit(1)

class SemanticRegenerator:
    """Classe pour régénérer proprement les analyses sémantiques"""
    
    def __init__(self, config_path=None):
        # Déterminer le chemin de configuration
        if config_path is None:
            # Chercher dans plusieurs emplacements
            possible_paths = [
                project_root / 'config' / 'config.yaml',
                parent_root / 'config' / 'config.yaml',
                Path('config/config.yaml'),
                Path('../config/config.yaml')
            ]
            
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            
            if config_path is None:
                logger.error("❌ Fichier de configuration non trouvé")
                sys.exit(1)
        
        logger.info(f"📋 Chargement de la configuration: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Charger le corpus
        logger.info("📥 Chargement du corpus...")
        self.corpus_loader = CorpusLoader(self.config)
        self.corpus = self.corpus_loader.load_corpus()
        
        # Initialiser le préprocesseur
        self.preprocessor = TextPreprocessor(self.config)
        
        logger.info(f"✅ Corpus chargé: {len(self.corpus['gaza'])} articles Gaza, "
                   f"{len(self.corpus['ukraine'])} articles Ukraine")
    
    def preprocess_corpus(self):
        """Prétraiter le corpus pour l'analyse sémantique"""
        logger.info("🔧 Prétraitement du corpus...")
        
        processed_corpus = {}
        for topic, articles in self.corpus.items():
            processed_texts = []
            for article in articles:
                if isinstance(article, dict) and 'content' in article:
                    text = article['content']
                elif isinstance(article, str):
                    text = article
                else:
                    continue
                
                # Nettoyer le texte
                cleaned = self.preprocessor.clean_text(text)
                tokens = self.preprocessor.tokenize(cleaned)
                processed_texts.append(' '.join(tokens))
            
            processed_corpus[topic] = processed_texts
        
        logger.info(f"✅ Corpus prétraité: {sum(len(v) for v in processed_corpus.values())} textes")
        return processed_corpus
    
    def generate_kwic(self, corpus, window_size=5):
        """Générer les concordances (KWIC)"""
        logger.info("🔍 Génération des concordances KWIC...")
        
        kwic_results = {}
        
        # Mots-clés à analyser pour chaque topic
        keywords_by_topic = {
            'gaza': ['attack', 'civilian', 'israel', 'hamas', 'war', 'death', 'violence', 'conflict'],
            'ukraine': ['attack', 'civilian', 'russian', 'ukrainian', 'war', 'invasion', 'resistance', 'freedom']
        }
        
        for topic, texts in corpus.items():
            topic_kwic = {}
            keywords = keywords_by_topic.get(topic, [])
            
            for keyword in keywords:
                concordances = []
                
                for text in texts:
                    words = text.lower().split()
                    
                    # Trouver les occurrences du mot-clé
                    for i, word in enumerate(words):
                        if keyword in word or word in keyword:
                            # Extraire le contexte
                            left_context = ' '.join(words[max(0, i - window_size):i])
                            right_context = ' '.join(words[i + 1:i + 1 + window_size])
                            
                            concordances.append({
                                'left': left_context,
                                'keyword': keyword,
                                'right': right_context,
                                'sentence': text[:200]  # Extraire un extrait
                            })
                
                if concordances:
                    topic_kwic[keyword] = {
                        'total_occurrences': len(concordances),
                        'examples': concordances[:10]  # Limiter à 10 exemples
                    }
            
            kwic_results[topic] = topic_kwic
        
        logger.info(f"✅ KWIC généré: {sum(len(v) for v in kwic_results.values())} mots-clés analysés")
        return kwic_results
    
    def generate_semantic_fields(self, corpus, top_n=20):
        """Générer les champs sémantiques (mots les plus fréquents)"""
        logger.info("🏷️ Génération des champs sémantiques...")
        
        from collections import Counter
        import re
        
        semantic_fields = {}
        
        for topic, texts in corpus.items():
            # Compter tous les mots
            all_words = []
            for text in texts:
                # Nettoyer et tokenizer
                words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
                all_words.extend(words)
            
            # Compter les fréquences
            word_counts = Counter(all_words)
            
            # Exclure les stopwords
            stopwords = set(['the', 'and', 'that', 'for', 'with', 'this', 'from', 'have', 'were', 'said'])
            filtered_counts = {word: count for word, count in word_counts.items() 
                             if word not in stopwords and count > 2}
            
            # Trier et prendre les top_n
            top_words = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
            
            semantic_fields[topic] = [
                {'word': word, 'count': count, 'score': count / len(all_words) * 100}
                for word, count in top_words
            ]
        
        logger.info("✅ Champs sémantiques générés")
        return semantic_fields
    
    def generate_word_clusters(self, corpus, n_clusters=4):
        """Générer des clusters sémantiques de mots"""
        logger.info("🔗 Génération des clusters sémantiques...")
        
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            import numpy as np
        except ImportError:
            logger.error("Scikit-learn non installé. Installation: pip install scikit-learn")
            return {}
        
        word_clusters = {}
        
        for topic, texts in corpus.items():
            # Vectoriser avec TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                X = vectorizer.fit_transform(texts)
                features = vectorizer.get_feature_names_out()
                
                # Appliquer K-means clustering
                kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
                kmeans.fit(X)
                
                # Organiser les mots par cluster
                clusters = {}
                for i in range(kmeans.n_clusters):
                    # Trouver les indices des mots dans ce cluster
                    cluster_indices = np.where(kmeans.labels_ == i)[0]
                    
                    # Pour chaque document dans le cluster, extraire les mots importants
                    cluster_words = []
                    for idx in cluster_indices[:10]:  # Limiter à 10 documents par cluster
                        # Obtenir les mots avec les scores TF-IDF les plus élevés
                        doc_vector = X[idx]
                        feature_indices = doc_vector.nonzero()[1]
                        doc_words = [(features[j], doc_vector[0, j]) for j in feature_indices]
                        doc_words.sort(key=lambda x: x[1], reverse=True)
                        
                        # Ajouter les mots uniques
                        for word, score in doc_words[:5]:
                            if word not in cluster_words:
                                cluster_words.append(word)
                    
                    clusters[f'cluster_{i}'] = cluster_words[:15]  # Limiter à 15 mots par cluster
                
                word_clusters[topic] = clusters
                
            except Exception as e:
                logger.warning(f"Erreur clustering pour {topic}: {e}")
                word_clusters[topic] = {}
        
        logger.info("✅ Clusters sémantiques générés")
        return word_clusters
    
    def generate_word2vec_similarities(self, corpus):
        """Générer les similarités Word2Vec"""
        logger.info("🧠 Génération des similarités Word2Vec...")
        
        try:
            from gensim.models import Word2Vec
            from gensim.models.phrases import Phrases, Phraser
        except ImportError:
            logger.error("Gensim non installé. Installation: pip install gensim")
            return {}
        
        word2vec_results = {}
        
        for topic, texts in corpus.items():
            # Préparer les phrases pour Word2Vec
            sentences = []
            for text in texts:
                # Tokenizer en phrases puis en mots
                import nltk
                try:
                    nltk.data.find('tokenizers/punkt')
                except LookupError:
                    nltk.download('punkt', quiet=True)
                
                from nltk.tokenize import sent_tokenize, word_tokenize
                
                for sent in sent_tokenize(text):
                    words = word_tokenize(sent.lower())
                    # Filtrer les mots courts et non alphabétiques
                    words = [w for w in words if len(w) > 2 and w.isalpha()]
                    if len(words) > 3:  # Phrases avec au moins 4 mots
                        sentences.append(words)
            
            if len(sentences) < 10:
                logger.warning(f"Pas assez de données pour Word2Vec sur {topic}")
                continue
            
            # Détecter les bigrammes
            phrases = Phrases(sentences, min_count=5, threshold=10)
            bigram = Phraser(phrases)
            sentences_bigram = [bigram[sent] for sent in sentences]
            
            # Entraîner le modèle Word2Vec
            model = Word2Vec(
                sentences_bigram,
                vector_size=100,
                window=5,
                min_count=5,
                workers=4,
                epochs=10
            )
            
            # Mots-clés à analyser
            if topic == 'gaza':
                keywords = ['militants', 'civilian', 'attack', 'war', 'violence', 'conflict']
            else:
                keywords = ['civilians', 'russian', 'ukrainian', 'invasion', 'resistance', 'freedom']
            
            topic_similarities = {}
            for keyword in keywords:
                if keyword in model.wv:
                    try:
                        # Obtenir les mots les plus similaires
                        similar_words = model.wv.most_similar(keyword, topn=8)
                        
                        # Formater les résultats
                        topic_similarities[keyword] = [
                            {'word': word, 'similarity': round(score, 4)}
                            for word, score in similar_words
                            if 0.3 < score < 0.95  # Filtrer les scores extrêmes
                        ]
                    except Exception as e:
                        logger.warning(f"Erreur similarité pour {keyword}: {e}")
                        topic_similarities[keyword] = []
            
            word2vec_results[topic] = topic_similarities
        
        logger.info("✅ Similarités Word2Vec générées")
        return word2vec_results
    
    def generate_actor_semantics(self, corpus):
        """Générer l'analyse sémantique par acteur"""
        logger.info("👥 Génération de l'analyse par acteur...")
        
        import re
        from collections import Counter
        
        # Définir les acteurs par conflit
        actors_by_topic = {
            'gaza': ['israeli', 'hamas', 'palestinian', 'civilian', 'militant', 'soldier'],
            'ukraine': ['russian', 'ukrainian', 'civilian', 'soldier', 'president', 'victim']
        }
        
        actor_results = {}
        
        for topic, texts in corpus.items():
            actors = actors_by_topic.get(topic, [])
            topic_actors = {}
            
            for actor in actors:
                actor_data = {
                    'mentions': 0,
                    'associated_words': [],
                    'contexts': []
                }
                
                # Analyser chaque texte
                for text in texts:
                    # Compter les mentions
                    pattern = re.compile(r'\b' + re.escape(actor) + r'\b', re.IGNORECASE)
                    mentions = len(pattern.findall(text))
                    
                    if mentions > 0:
                        actor_data['mentions'] += mentions
                        
                        # Extraire le contexte
                        sentences = text.split('.')
                        for sent in sentences:
                            if actor.lower() in sent.lower():
                                # Tokenizer la phrase
                                words = re.findall(r'\b[a-zA-Z]{3,}\b', sent.lower())
                                # Exclure l'acteur lui-même et les stopwords
                                stopwords = {'the', 'and', 'that', 'for', 'with'}
                                context_words = [w for w in words 
                                               if w != actor.lower() and w not in stopwords]
                                actor_data['associated_words'].extend(context_words[:5])
                                
                                # Ajouter le contexte
                                if len(actor_data['contexts']) < 3:  # Limiter à 3 contextes
                                    actor_data['contexts'].append(sent[:150])
                
                # Analyser les mots associés
                if actor_data['associated_words']:
                    word_counts = Counter(actor_data['associated_words'])
                    top_words = word_counts.most_common(10)
                    actor_data['top_associated'] = [
                        {'word': word, 'count': count}
                        for word, count in top_words
                    ]
                
                topic_actors[actor] = actor_data
            
            actor_results[topic] = topic_actors
        
        logger.info("✅ Analyse par acteur générée")
        return actor_results
    
    def generate_topic_modeling(self, corpus, n_topics=3):
        """Générer le topic modeling avec LDA"""
        logger.info("📊 Génération du topic modeling (LDA)...")
        
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.decomposition import LatentDirichletAllocation
        except ImportError:
            logger.error("Scikit-learn non installé. Installation: pip install scikit-learn")
            return {}
        
        topic_results = {}
        
        for topic_name, texts in corpus.items():
            # Vectoriser avec CountVectorizer
            vectorizer = CountVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            try:
                X = vectorizer.fit_transform(texts[:50])  # Limiter à 50 textes pour la stabilité
                feature_names = vectorizer.get_feature_names_out()
                
                # Appliquer LDA
                lda = LatentDirichletAllocation(
                    n_components=min(n_topics, len(texts)),
                    random_state=42,
                    max_iter=10
                )
                lda.fit(X)
                
                # Extraire les topics
                topics_data = {}
                for topic_idx, topic in enumerate(lda.components_):
                    # Obtenir les mots les plus importants pour ce topic
                    top_words_idx = topic.argsort()[:-10 - 1:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    
                    # Déterminer la catégorie dominante
                    categories = {
                        'military': ['attack', 'war', 'soldier', 'military', 'force'],
                        'civilian': ['civilian', 'people', 'family', 'home', 'child'],
                        'political': ['government', 'president', 'official', 'state', 'country'],
                        'humanitarian': ['aid', 'help', 'support', 'crisis', 'need']
                    }
                    
                    # Calculer le score pour chaque catégorie
                    category_scores = {}
                    for category, keywords in categories.items():
                        score = sum(1 for word in top_words if any(kw in word for kw in keywords))
                        category_scores[category] = score
                    
                    dominant_category = max(category_scores.items(), key=lambda x: x[1])[0]
                    
                    topics_data[f'topic_{topic_idx}'] = {
                        'words': top_words,
                        'dominant_category': dominant_category,
                        'category_scores': category_scores
                    }
                
                topic_results[topic_name] = topics_data
                
            except Exception as e:
                logger.warning(f"Erreur LDA pour {topic_name}: {e}")
                topic_results[topic_name] = {}
        
        logger.info("✅ Topic modeling généré")
        return topic_results
    
    def run_complete_analysis(self):
        """Exécuter l'analyse sémantique complète"""
        logger.info("=" * 60)
        logger.info("🚀 DÉMARRAGE DE L'ANALYSE SÉMANTIQUE COMPLÈTE")
        logger.info("=" * 60)
        
        # 1. Prétraiter le corpus
        processed_corpus = self.preprocess_corpus()
        
        # 2. Générer toutes les analyses
        results = {
            'semantic_fields': self.generate_semantic_fields(processed_corpus),
            'kwic': self.generate_kwic(processed_corpus),
            'word_clusters': self.generate_word_clusters(processed_corpus),
            'word2vec': self.generate_word2vec_similarities(processed_corpus),
            'actor_semantics': self.generate_actor_semantics(processed_corpus),
            'topic_modeling': self.generate_topic_modeling(processed_corpus),
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'corpus_size': {k: len(v) for k, v in processed_corpus.items()},
                'analysis_types': ['semantic_fields', 'kwic', 'word_clusters', 
                                  'word2vec', 'actor_semantics', 'topic_modeling']
            }
        }
        
        # 3. Sauvegarder les résultats
        output_dir = project_root / 'analysis_results'
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"semantic_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Analyse sémantique sauvegardée dans: {output_file}")
        
        # 4. Vérifier la qualité des données
        self.verify_results_quality(results)
        
        return results
    
    def verify_results_quality(self, results):
        """Vérifier la qualité des résultats générés"""
        logger.info("\n🔍 VÉRIFICATION DE LA QUALITÉ DES DONNÉES")
        logger.info("-" * 40)
        
        issues = []
        
        # Vérifier Word2Vec
        if 'word2vec' in results:
            for topic, words in results['word2vec'].items():
                for word, similarities in words.items():
                    if similarities:
                        for sim in similarities:
                            if not (0 <= sim['similarity'] <= 1):
                                issues.append(f"Word2Vec {topic}.{word}: score invalide {sim['similarity']}")
                            if not isinstance(sim['word'], str) or len(sim['word']) < 2:
                                issues.append(f"Word2Vec {topic}.{word}: mot invalide")
        
        # Vérifier actor_semantics
        if 'actor_semantics' in results:
            for topic, actors in results['actor_semantics'].items():
                for actor, data in actors.items():
                    if 'associated_words' in data:
                        for word in data.get('associated_words', []):
                            if not isinstance(word, str):
                                issues.append(f"Actor {topic}.{actor}: mot non-string")
        
        if issues:
            logger.warning(f"⚠️  {len(issues)} problèmes détectés:")
            for issue in issues[:5]:  # Afficher seulement les 5 premiers
                logger.warning(f"  - {issue}")
        else:
            logger.info("✅ Toutes les données semblent valides")
        
        # Afficher des statistiques
        logger.info("\n📊 STATISTIQUES DE L'ANALYSE:")
        for key, value in results.items():
            if key != 'metadata':
                if isinstance(value, dict):
                    logger.info(f"  {key}: {len(value)} topics")
                    for subkey, subvalue in value.items():
                        if isinstance(subvalue, dict):
                            logger.info(f"    {subkey}: {len(subvalue)} éléments")

def main():
    """Fonction principale"""
    try:
        # Initialiser le régénérateur
        regenerator = SemanticRegenerator()
        
        # Exécuter l'analyse complète
        results = regenerator.run_complete_analysis()
        
        print("\n" + "=" * 60)
        print("🎉 ANALYSE SÉMANTIQUE COMPLÈTE TERMINÉE !")
        print("=" * 60)
        
        # Afficher un résumé
        print("\n📋 RÉSUMÉ DES DONNÉES GÉNÉRÉES:")
        for key, value in results.items():
            if key != 'metadata':
                if isinstance(value, dict):
                    print(f"\n{key.upper()}:")
                    for topic, data in value.items():
                        size = len(data) if isinstance(data, (dict, list)) else 'N/A'
                        print(f"  {topic}: {size} éléments")
        
        print("\n✅ Prêt pour l'interface web!")
        print("   Lancez: python main.py --web")
        
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'analyse: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())