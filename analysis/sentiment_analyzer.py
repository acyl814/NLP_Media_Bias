"""
Analyseur de sentiment pour comparer la tonalité entre les conflits
Utilise plusieurs modèles de sentiment pour une analyse robuste
"""

import json
import os
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd
import numpy as np
import yaml
from tqdm import tqdm
import logging
import re

# Sentiment Analysis Libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False
    print("TextBlob non installé. Installation: pip install textblob")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VADER non installé. Installation: pip install vaderSentiment")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers non installé. Installation: pip install transformers torch")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyseur de sentiment multi-modèles pour la détection de biais"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialise l'analyseur de sentiment
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sentiment_config = self.config['analysis']['sentiment']
        self.bias_config = self.config['analysis']['bias_detection']
        
        # Modèles à utiliser
        self.models = self.sentiment_config.get('models', ['textblob', 'vader'])
        
        # Initialiser les modèles
        self.analyzers = {}
        self._initialize_analyzers()
        
        # Corpus
        self.corpus = []
        self.topic_articles = {}
        
        # Mots-cibles étendus pour l'analyse approfondie des acteurs
        self.target_words = {
            'gaza': [
                # Acteurs principaux
                'israel', 'israeli', 'idf', 'israelis', 'israel\'s',
                'palestinians', 'palestinian', 'gaza', 'gazan', 'gaza\'s',
                'hamas', 'militants', 'fighters', 'terrorists', 'extremists',
                'civilians', 'victims', 'children', 'families', 'women',
                'forces', 'soldiers', 'army', 'military', 'troops',
                'government', 'authorities', 'leaders', 'officials',
                'residents', 'people', 'population', 'community'
            ],
            'ukraine': [
                # Acteurs principaux
                'ukraine', 'ukrainian', 'ukrainians', 'ukraine\'s',
                'russia', 'russian', 'russians', 'russia\'s',
                'zelensky', 'putin',
                'civilians', 'victims', 'children', 'families', 'women',
                'heroes', 'soldiers', 'forces', 'military', 'troops',
                'nato', 'allies', 'west', 'european',
                'government', 'authorities', 'leaders', 'officials',
                'invaders', 'aggressors', 'occupiers',
                'resistance', 'defenders', 'volunteers'
            ]
        }
        
        # Résultats
        self.results = {}
    
    def _initialize_analyzers(self):
        """Initialise les analyseurs de sentiment"""
        
        for model_name in self.models:
            try:
                if model_name == 'textblob' and TEXTBLOB_AVAILABLE:
                    self.analyzers['textblob'] = None  # TextBlob est stateless
                    logger.info("TextBlob initialisé")
                
                elif model_name == 'vader' and VADER_AVAILABLE:
                    self.analyzers['vader'] = SentimentIntensityAnalyzer()
                    logger.info("VADER initialisé")
                
                elif model_name == 'transformers' and TRANSFORMERS_AVAILABLE:
                    # Utiliser un modèle BERT pour le sentiment
                    self.analyzers['transformers'] = pipeline(
                        "sentiment-analysis",
                        model="nlptown/bert-base-multilingual-uncased-sentiment"
                    )
                    logger.info("Transformers (BERT) initialisé")
            
            except Exception as e:
                logger.warning(f"Impossible d'initialiser {model_name}: {e}")
    
    def load_corpus(self, corpus_path: str):
        """
        Charge le corpus prétraité
        
        Args:
            corpus_path: Chemin vers le fichier corpus
        """
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            self.corpus = json.load(f)
        
        # Organiser par topic
        for article in self.corpus:
            topic = article.get('topic', 'unknown')
            
            if topic not in self.topic_articles:
                self.topic_articles[topic] = []
            
            self.topic_articles[topic].append(article)
        
        logger.info(f"Corpus chargé: {len(self.corpus)} articles")
        logger.info(f"Topics: {list(self.topic_articles.keys())}")
    
    def analyze_all(self, output_dir="analysis_results") -> Dict:
        """
        Effectue toutes les analyses de sentiment
        
        Args:
            output_dir: Répertoire de sortie
            
        Returns:
            Résultats de toutes les analyses
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Début de l'analyse de sentiment...")
        
        # 1. Analyse globale par article
        self.results['article_sentiment'] = self.analyze_article_sentiment()
        
        # 2. Analyse par topic
        self.results['topic_sentiment'] = self.analyze_topic_sentiment()
        
        # 3. Analyse approfondie par acteur/mot-cible
        self.results['target_word_sentiment'] = self.analyze_target_word_sentiment()
        
        # 4. Analyse des segments
        self.results['segment_sentiment'] = self.analyze_segments()
        
        # 5. Comparaison entre topics
        self.results['sentiment_comparison'] = self.compare_sentiment()
        
        # 6. Analyse temporelle (si dates disponibles)
        self.results['temporal_sentiment'] = self.analyze_temporal_sentiment()
        
        # 7. Analyse comparative détaillée des acteurs
        self.results['actor_comparative_analysis'] = self.analyze_actor_comparisons()
        
        # Sauvegarder les résultats
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(output_dir, f"sentiment_analysis_{timestamp}.json")
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Analyses sauvegardées dans: {results_path}")
        
        # Générer des CSV pour l'exploration
        self._export_to_csv(output_dir, timestamp)
        
        return self.results
    
    def analyze_article_sentiment(self) -> List[Dict]:
        """
        Analyse le sentiment de chaque article
        
        Returns:
            Sentiment par article
        """
        
        article_sentiments = []
        
        for article in tqdm(self.corpus, desc="Analyzing article sentiment"):
            content = article.get('content_clean', article.get('content', ''))
            
            if not content:
                continue
            
            # Analyser avec chaque modèle
            sentiments = {}
            
            for model_name, analyzer in self.analyzers.items():
                try:
                    if model_name == 'textblob':
                        blob = TextBlob(content)
                        polarity = blob.sentiment.polarity
                        subjectivity = blob.sentiment.subjectivity
                        
                        sentiments['textblob'] = {
                            'polarity': float(polarity),
                            'subjectivity': float(subjectivity),
                            'label': self._textblob_to_label(polarity, subjectivity)
                        }
                    
                    elif model_name == 'vader':
                        scores = analyzer.polarity_scores(content)
                        
                        sentiments['vader'] = {
                            'compound': float(scores['compound']),
                            'positive': float(scores['pos']),
                            'negative': float(scores['neg']),
                            'neutral': float(scores['neu']),
                            'label': self._vader_to_label(scores['compound'])
                        }
                    
                    elif model_name == 'transformers':
                        # Analyser par segments pour éviter les limites de taille
                        segments = self._split_text(content, max_length=500)
                        segment_results = []
                        
                        for segment in segments:
                            result = analyzer(segment)[0]
                            segment_results.append(result)
                        
                        # Moyenner les résultats
                        avg_label = max(set([r['label'] for r in segment_results]), 
                                      key=[r['label'] for r in segment_results].count)
                        avg_score = float(np.mean([r['score'] for r in segment_results]))
                        
                        sentiments['transformers'] = {
                            'label': avg_label,
                            'score': avg_score
                        }
                
                except Exception as e:
                    logger.warning(f"Erreur avec {model_name}: {e}")
                    continue
            
            # Calculer un consensus
            consensus = self._compute_consensus(sentiments)
            
            article_sentiments.append({
                'id': article.get('id', ''),
                'title': article.get('title', ''),
                'topic': article.get('topic', 'unknown'),
                'source': article.get('source', 'unknown'),
                'word_count': len(content.split()),
                'sentiments': sentiments,
                'consensus': consensus
            })
        
        logger.info(f"Sentiment analysé pour {len(article_sentiments)} articles")
        return article_sentiments
    
    def analyze_topic_sentiment(self) -> Dict[str, Dict]:
        """
        Analyse le sentiment agrégé par topic
        
        Returns:
            Sentiment par topic
        """
        
        topic_sentiment = {}
        
        for topic, articles in self.topic_articles.items():
            # Récupérer les sentiments des articles de ce topic
            topic_articles_sentiment = [
                art for art in self.results['article_sentiment'] 
                if art['topic'] == topic
            ]
            
            if not topic_articles_sentiment:
                continue
            
            # Agréger par modèle
            aggregated = {}
            
            for model_name in self.models:
                if model_name not in self.analyzers:
                    continue
                
                model_results = []
                
                for article_sent in topic_articles_sentiment:
                    if model_name in article_sent['sentiments']:
                        model_results.append(article_sent['sentiments'][model_name])
                
                if not model_results:
                    continue
                
                if model_name == 'textblob':
                    aggregated['textblob'] = {
                        'avg_polarity': float(np.mean([r['polarity'] for r in model_results])),
                        'avg_subjectivity': float(np.mean([r['subjectivity'] for r in model_results])),
                        'distribution': dict(Counter([r['label'] for r in model_results]))
                    }
                
                elif model_name == 'vader':
                    aggregated['vader'] = {
                        'avg_compound': float(np.mean([r['compound'] for r in model_results])),
                        'avg_positive': float(np.mean([r['positive'] for r in model_results])),
                        'avg_negative': float(np.mean([r['negative'] for r in model_results])),
                        'avg_neutral': float(np.mean([r['neutral'] for r in model_results])),
                        'distribution': dict(Counter([r['label'] for r in model_results]))
                    }
                
                elif model_name == 'transformers':
                    aggregated['transformers'] = {
                        'distribution': dict(Counter([r['label'] for r in model_results])),
                        'avg_score': float(np.mean([r['score'] for r in model_results]))
                    }
            
            # Consensus global
            consensus_labels = [art['consensus']['label'] for art in topic_articles_sentiment]
            consensus_scores = [art['consensus']['score'] for art in topic_articles_sentiment]
            
            topic_sentiment[topic] = {
                'total_articles': len(topic_articles_sentiment),
                'aggregated_sentiment': aggregated,
                'consensus_distribution': dict(Counter(consensus_labels)),
                'avg_consensus_score': float(np.mean(consensus_scores) if consensus_scores else 0),
                'sentiment_ratio': self._compute_sentiment_ratio(consensus_labels)
            }
        
        logger.info("Sentiment par topic analysé")
        return topic_sentiment
    
    def analyze_target_word_sentiment(self) -> Dict[str, Dict[str, List[Dict]]]:
        """
        Analyse approfondie du sentiment autour des acteurs/mots-cibles
        
        Returns:
            Sentiment par acteur et topic
        """
        
        target_sentiment = {}
        
        for topic, articles in tqdm(self.topic_articles.items(), desc="Analyzing topics"):
            target_sentiment[topic] = {}
            target_words = self.target_words.get(topic, [])
            
            for target_word in tqdm(target_words, desc=f"  {topic} actors", leave=False):
                contexts = []
                
                for article in articles:
                    content = article.get('content_clean', article.get('content', '')).lower()
                    
                    # Vérifier si l'article mentionne le mot-cible
                    if target_word.lower() in content:
                        # Diviser en phrases si non disponible
                        sentences = article.get('sentences', [])
                        if not sentences:
                            # Fallback: split by punctuation
                            sentences = re.split(r'[.!?]+', article.get('content', ''))
                        
                        # Trouver toutes les phrases avec le mot-cible
                        for sentence in sentences:
                            if target_word.lower() in sentence.lower():
                                # Analyser le sentiment spécifique de cette phrase
                                sentence_sentiment = self._analyze_sentence_sentiment(sentence)
                                
                                # Identifier le contexte (mots environnants)
                                words = sentence.lower().split()
                                try:
                                    # Trouver toutes les occurrences
                                    indices = [i for i, word in enumerate(words) 
                                              if word == target_word.lower() or 
                                              word.startswith(target_word.lower())]
                                    
                                    if indices:
                                        idx = indices[0]
                                        start = max(0, idx - 3)
                                        end = min(len(words), idx + 4)
                                        context_window = ' '.join(words[start:end])
                                    else:
                                        context_window = sentence[:100] + "..."
                                except:
                                    context_window = sentence[:100] + "..."
                                
                                contexts.append({
                                    'sentence': sentence,
                                    'context_window': context_window,
                                    'sentiment': sentence_sentiment,
                                    'source': article.get('source', 'unknown'),
                                    'title': article.get('title', ''),
                                    'article_id': article.get('id', ''),
                                    'word_position': self._find_word_position(sentence, target_word)
                                })
                
                # Agréger et calculer les statistiques détaillées
                if contexts:
                    aggregated = self._aggregate_context_sentiment_detailed(contexts, target_word)
                    target_sentiment[topic][target_word] = {
                        'contexts': contexts[:20],  # Limiter pour la taille du fichier
                        'aggregated': aggregated,
                        'statistics': self._compute_word_statistics(contexts, topic, target_word)
                    }
        
        logger.info("Analyse approfondie des acteurs terminée")
        return target_sentiment
    
    def analyze_segments(self, segment_size: int = 100) -> List[Dict]:
        """
        Analyse le sentiment par segments de texte
        
        Args:
            segment_size: Taille des segments en mots
            
        Returns:
            Sentiment par segment
        """
        
        segment_sentiments = []
        
        for article in tqdm(self.corpus, desc="Analyzing segments"):
            content = article.get('content_clean', article.get('content', ''))
            
            if not content:
                continue
            
            # Diviser en segments
            words = content.split()
            segments = [
                ' '.join(words[i:i+segment_size]) 
                for i in range(0, len(words), segment_size)
            ]
            
            # Analyser chaque segment
            segment_results = []
            for i, segment in enumerate(segments):
                segment_sent = self._analyze_sentence_sentiment(segment)
                segment_results.append({
                    'segment_id': i,
                    'word_count': len(segment.split()),
                    'sentiment': segment_sent
                })
            
            segment_sentiments.append({
                'article_id': article.get('id', ''),
                'topic': article.get('topic', 'unknown'),
                'segments': segment_results
            })
        
        logger.info(f"Sentiment analysé pour les segments")
        return segment_sentiments
    
    def compare_sentiment(self) -> Dict[str, Dict]:
        """
        Compare le sentiment entre les topics
        
        Returns:
            Comparaison des sentiments
        """
        
        comparisons = {}
        
        # Récupérer les distributions de sentiment
        topic_distributions = {}
        for topic, data in self.results['topic_sentiment'].items():
            topic_distributions[topic] = data['consensus_distribution']
        
        # Comparer les distributions
        topics = list(topic_distributions.keys())
        
        for i, topic1 in enumerate(topics):
            for topic2 in topics[i+1:]:
                comparison_key = f"{topic1}_vs_{topic2}"
                
                # Calculer les différences
                all_labels = set(topic_distributions[topic1].keys()) | set(topic_distributions[topic2].keys())
                
                diff_data = {}
                for label in all_labels:
                    count1 = topic_distributions[topic1].get(label, 0)
                    count2 = topic_distributions[topic2].get(label, 0)
                    
                    total1 = sum(topic_distributions[topic1].values())
                    total2 = sum(topic_distributions[topic2].values())
                    
                    pct1 = (count1 / total1 * 100) if total1 > 0 else 0
                    pct2 = (count2 / total2 * 100) if total2 > 0 else 0
                    
                    diff_data[label] = {
                        f'{topic1}_count': count1,
                        f'{topic2}_count': count2,
                        f'{topic1}_percent': float(pct1),
                        f'{topic2}_percent': float(pct2),
                        'difference': float(pct1 - pct2)
                    }
                
                comparisons[comparison_key] = diff_data
        
        logger.info("Comparaison de sentiment complétée")
        return comparisons
    
    def analyze_temporal_sentiment(self) -> Dict[str, Dict]:
        """
        Analyse l'évolution temporelle du sentiment
        
        Returns:
            Sentiment temporel par topic
        """
        
        temporal_sentiment = {}
        
        for topic, articles in self.topic_articles.items():
            # Grouper par date (mois-année)
            monthly_sentiment = defaultdict(list)
            
            for article in articles:
                publish_date = article.get('publish_date')
                
                if not publish_date:
                    continue
                
                # Extraire le mois-année
                try:
                    date_obj = pd.to_datetime(publish_date)
                    month_year = date_obj.strftime('%Y-%m')
                    
                    # Trouver le sentiment de l'article
                    article_sent = next(
                        (art for art in self.results['article_sentiment'] 
                         if art['id'] == article.get('id', '')),
                        None
                    )
                    
                    if article_sent:
                        monthly_sentiment[month_year].append(article_sent['consensus'])
                
                except Exception as e:
                    continue
            
            # Agréger par mois
            monthly_aggregated = {}
            for month, sentiments in monthly_sentiment.items():
                labels = [s['label'] for s in sentiments]
                scores = [s['score'] for s in sentiments]
                
                monthly_aggregated[month] = {
                    'total_articles': len(sentiments),
                    'sentiment_distribution': dict(Counter(labels)),
                    'avg_score': float(np.mean(scores) if scores else 0)
                }
            
            temporal_sentiment[topic] = monthly_aggregated
        
        logger.info("Analyse temporelle du sentiment complétée")
        return temporal_sentiment
    
    def analyze_actor_comparisons(self) -> Dict:
        """
        Analyse comparative approfondie des acteurs entre les conflits
        
        Returns:
            Comparaisons détaillées entre acteurs similaires
        """
        
        comparisons = {}
        
        # Paires d'acteurs comparables entre Gaza et Ukraine
        comparable_actors = {
            'civilians': ('civilians', 'civilians'),
            'military_forces': ('forces', 'forces'),
            'government': ('government', 'government'),
            'leaders': ('hamas', 'zelensky'),  # Simplification
            'victims': ('victims', 'victims')
        }
        
        for comparison_name, (actor_gaza, actor_ukraine) in comparable_actors.items():
            comparison_data = {}
            
            # Données pour Gaza
            if 'gaza' in self.results.get('target_word_sentiment', {}):
                gaza_actors = self.results['target_word_sentiment']['gaza']
                # Chercher l'acteur correspondant (avec variantes)
                gaza_data = None
                for actor, data in gaza_actors.items():
                    if actor_gaza in actor or actor in actor_gaza:
                        gaza_data = data
                        break
                
                if gaza_data:
                    comparison_data['gaza'] = {
                        'actor': actor,
                        'dominant_sentiment': gaza_data['aggregated']['dominant_sentiment'],
                        'distribution': gaza_data['aggregated']['distribution'],
                        'total_contexts': gaza_data['aggregated']['total_contexts']
                    }
            
            # Données pour Ukraine
            if 'ukraine' in self.results.get('target_word_sentiment', {}):
                ukraine_actors = self.results['target_word_sentiment']['ukraine']
                ukraine_data = None
                for actor, data in ukraine_actors.items():
                    if actor_ukraine in actor or actor in actor_ukraine:
                        ukraine_data = data
                        break
                
                if ukraine_data:
                    comparison_data['ukraine'] = {
                        'actor': actor,
                        'dominant_sentiment': ukraine_data['aggregated']['dominant_sentiment'],
                        'distribution': ukraine_data['aggregated']['distribution'],
                        'total_contexts': ukraine_data['aggregated']['total_contexts']
                    }
            
            # Calculer les différences si les deux sont présents
            if 'gaza' in comparison_data and 'ukraine' in comparison_data:
                gaza_dist = comparison_data['gaza']['distribution']
                ukraine_dist = comparison_data['ukraine']['distribution']
                
                # Calculer les pourcentages
                gaza_total = sum(gaza_dist.values())
                ukraine_total = sum(ukraine_dist.values())
                
                percentages = {}
                for label in set(gaza_dist.keys()) | set(ukraine_dist.keys()):
                    gaza_pct = (gaza_dist.get(label, 0) / gaza_total * 100) if gaza_total > 0 else 0
                    ukraine_pct = (ukraine_dist.get(label, 0) / ukraine_total * 100) if ukraine_total > 0 else 0
                    
                    percentages[label] = {
                        'gaza_percent': float(gaza_pct),
                        'ukraine_percent': float(ukraine_pct),
                        'difference': float(gaza_pct - ukraine_pct),
                        'bias_direction': 'gaza' if gaza_pct > ukraine_pct else 'ukraine'
                    }
                
                comparison_data['comparison'] = percentages
            
            comparisons[comparison_name] = comparison_data
        
        return comparisons
    
    def _analyze_sentence_sentiment(self, text: str) -> Dict:
        """
        Analyse le sentiment d'un texte court
        
        Args:
            text: Texte à analyser
            
        Returns:
            Résultats de sentiment
        """
        
        results = {}
        
        # TextBlob
        if 'textblob' in self.analyzers and TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                results['textblob'] = {
                    'polarity': float(polarity),
                    'subjectivity': float(subjectivity),
                    'label': self._textblob_to_label(polarity, subjectivity)
                }
            except Exception as e:
                logger.debug(f"TextBlob error: {e}")
        
        # VADER
        if 'vader' in self.analyzers and VADER_AVAILABLE:
            try:
                analyzer = self.analyzers['vader']
                scores = analyzer.polarity_scores(text)
                
                results['vader'] = {
                    'compound': float(scores['compound']),
                    'positive': float(scores['pos']),
                    'negative': float(scores['neg']),
                    'neutral': float(scores['neu']),
                    'label': self._vader_to_label(scores['compound'])
                }
            except Exception as e:
                logger.debug(f"VADER error: {e}")
        
        # Consensus
        consensus = self._compute_consensus(results)
        
        return {
            'results': results,
            'consensus': consensus
        }
    
    def _find_word_position(self, sentence: str, target_word: str) -> str:
        """Identifie la position syntaxique approximative du mot-cible"""
        
        words = sentence.lower().split()
        target_word_lower = target_word.lower()
        
        # Chercher le mot (exact ou partiel)
        for i, word in enumerate(words):
            if target_word_lower in word or word in target_word_lower:
                total_words = len(words)
                
                if i == 0:
                    return "beginning"
                elif i == total_words - 1:
                    return "end"
                elif i < total_words / 3:
                    return "early"
                elif i > 2 * total_words / 3:
                    return "late"
                else:
                    return "middle"
        
        return "unknown"
    
    def _aggregate_context_sentiment_detailed(self, contexts: List[Dict], target_word: str) -> Dict:
        """Agrège le sentiment avec plus de détails"""
        
        all_labels = []
        all_scores = []
        by_source = defaultdict(list)
        by_position = defaultdict(list)
        
        for context in contexts:
            consensus = context['sentiment']['consensus']
            all_labels.append(consensus['label'])
            all_scores.append(consensus['score'])
            
            # Regrouper par source
            source = context['source']
            by_source[source].append(consensus['label'])
            
            # Regrouper par position
            position = context.get('word_position', 'unknown')
            by_position[position].append(consensus['label'])
        
        # Distribution générale
        distribution = dict(Counter(all_labels))
        
        # Distribution par source
        source_distribution = {}
        for source, labels in by_source.items():
            source_distribution[source] = dict(Counter(labels))
        
        # Distribution par position
        position_distribution = {}
        for position, labels in by_position.items():
            position_distribution[position] = dict(Counter(labels))
        
        # Scores
        avg_score = float(np.mean(all_scores) if all_scores else 0)
        
        # Sentiment dominant
        if all_labels:
            dominant_label = max(set(all_labels), key=all_labels.count)
        else:
            dominant_label = 'neutral'
        
        # Calculer la force du sentiment
        sentiment_strength = self._compute_sentiment_strength(all_labels, all_scores)
        
        return {
            'dominant_sentiment': dominant_label,
            'distribution': distribution,
            'total_contexts': len(contexts),
            'avg_confidence': avg_score,
            'source_distribution': source_distribution,
            'position_distribution': position_distribution,
            'sentiment_strength': sentiment_strength,
            'unique_articles': len(set([c['article_id'] for c in contexts]))
        }
    
    def _compute_word_statistics(self, contexts: List[Dict], topic: str, target_word: str) -> Dict:
        """Calcule des statistiques spécifiques pour le mot-cible"""
        
        # Compter les co-occurrences avec d'autres mots
        cooccurring_words = defaultdict(int)
        sentiment_adjacents = defaultdict(list)
        
        for context in contexts:
            sentence = context['sentence'].lower()
            words = sentence.split()
            
            # Trouver les mots adjacents
            try:
                # Chercher toutes les occurrences du mot
                for i, word in enumerate(words):
                    if target_word.lower() in word or word in target_word.lower():
                        # Mots avant et après
                        for j in range(max(0, i-2), min(len(words), i+3)):
                            if j != i and len(words[j]) > 2:  # Ignorer les mots courts
                                adjacent_word = words[j]
                                cooccurring_words[adjacent_word] += 1
                                
                                # Associer le sentiment au mot adjacent
                                sentiment = context['sentiment']['consensus']['label']
                                sentiment_adjacents[adjacent_word].append(sentiment)
            except Exception as e:
                continue
        
        # Top 10 des co-occurrences
        top_cooccurrences = sorted(cooccurring_words.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Analyser les modificateurs fréquents
        modifiers = {}
        for word, count in top_cooccurrences:
            if word in sentiment_adjacents:
                sentiment_dist = dict(Counter(sentiment_adjacents[word]))
                if sentiment_dist:
                    dominant = max(sentiment_dist.items(), key=lambda x: x[1])
                    modifiers[word] = {
                        'count': count,
                        'sentiment_distribution': sentiment_dist,
                        'dominant_sentiment': dominant[0],
                        'dominant_count': dominant[1]
                    }
        
        return {
            'top_cooccurrences': dict(top_cooccurrences),
            'modifier_analysis': modifiers,
            'context_diversity': len(set([c['source'] for c in contexts])),
            'avg_context_length': float(np.mean([len(c['sentence'].split()) for c in contexts]) if contexts else 0)
        }
    
    def _compute_sentiment_strength(self, labels: List[str], scores: List[float]) -> Dict:
        """Calcule la force et la cohérence du sentiment"""
        
        if not labels:
            return {'strength': 'weak', 'dominance_ratio': 0.0, 'polarization': 0.0}
        
        # Distribution
        dist = Counter(labels)
        total = len(labels)
        
        # Force (dominance du sentiment principal)
        max_count = max(dist.values())
        strength_ratio = max_count / total
        
        if strength_ratio > 0.7:
            strength_label = 'very_strong'
        elif strength_ratio > 0.5:
            strength_label = 'strong'
        elif strength_ratio > 0.3:
            strength_label = 'moderate'
        else:
            strength_label = 'weak'
        
        # Polarisation (mix de positifs et négatifs)
        positive_pct = dist.get('positive', 0) / total
        negative_pct = dist.get('negative', 0) / total
        polarization_score = abs(positive_pct - negative_pct)
        
        return {
            'strength': strength_label,
            'dominance_ratio': float(strength_ratio),
            'polarization': float(polarization_score)
        }
    
    def _compute_consensus(self, sentiments: Dict) -> Dict:
        """
        Calcule un consensus entre les modèles
        
        Args:
            sentiments: Sentiments par modèle
            
        Returns:
            Consensus
        """
        
        if not sentiments:
            return {'label': 'neutral', 'score': 0.0}
        
        # Collecter les labels
        labels = []
        
        for model_name, result in sentiments.items():
            if model_name == 'textblob':
                labels.append(result['label'])
            elif model_name == 'vader':
                labels.append(result['label'])
            elif model_name == 'transformers':
                labels.append(result['label'])
        
        # Vote majoritaire
        if labels:
            consensus_label = max(set(labels), key=labels.count)
            confidence = labels.count(consensus_label) / len(labels)
        else:
            consensus_label = 'neutral'
            confidence = 0.0
        
        return {
            'label': consensus_label,
            'score': float(confidence)
        }
    
    def _textblob_to_label(self, polarity: float, subjectivity: float) -> str:
        """
        Convertit les scores TextBlob en label
        
        Args:
            polarity: Polarité (-1 à 1)
            subjectivity: Subjectivité (0 à 1)
            
        Returns:
            Label de sentiment
        """
        
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _vader_to_label(self, compound: float) -> str:
        """
        Convertit le score VADER en label
        
        Args:
            compound: Score compound (-1 à 1)
            
        Returns:
            Label de sentiment
        """
        
        if compound >= 0.05:
            return 'positive'
        elif compound <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def _compute_sentiment_ratio(self, labels: List[str]) -> Dict[str, float]:
        """
        Calcule les ratios de sentiment
        
        Args:
            labels: Liste de labels de sentiment
            
        Returns:
            Ratios de sentiment
        """
        
        total = len(labels)
        if total == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        counts = Counter(labels)
        
        return {
            'positive': float(counts.get('positive', 0) / total),
            'negative': float(counts.get('negative', 0) / total),
            'neutral': float(counts.get('neutral', 0) / total)
        }
    
    def _split_text(self, text: str, max_length: int = 500) -> List[str]:
        """
        Divise un texte en segments
        
        Args:
            text: Texte à diviser
            max_length: Longueur maximale des segments
            
        Returns:
            Liste de segments
        """
        
        words = text.split()
        segments = []
        
        for i in range(0, len(words), max_length):
            segment = ' '.join(words[i:i+max_length])
            segments.append(segment)
        
        return segments
    
    def _export_to_csv(self, output_dir: str, timestamp: str):
        """
        Exporte les résultats en CSV
        
        Args:
            output_dir: Répertoire de sortie
            timestamp: Timestamp pour les noms de fichiers
        """
        
        # Sentiment par article
        if self.results['article_sentiment']:
            flat_data = []
            for art in self.results['article_sentiment']:
                row = {
                    'id': art['id'],
                    'title': art['title'],
                    'topic': art['topic'],
                    'source': art['source'],
                    'word_count': art['word_count'],
                    'consensus_label': art['consensus']['label'],
                    'consensus_score': art['consensus']['score']
                }
                
                # Ajouter les scores de chaque modèle
                for model_name, sentiment in art['sentiments'].items():
                    if model_name == 'textblob':
                        row['textblob_polarity'] = sentiment['polarity']
                        row['textblob_subjectivity'] = sentiment['subjectivity']
                        row['textblob_label'] = sentiment['label']
                    elif model_name == 'vader':
                        row['vader_compound'] = sentiment['compound']
                        row['vader_positive'] = sentiment['positive']
                        row['vader_negative'] = sentiment['negative']
                        row['vader_neutral'] = sentiment['neutral']
                        row['vader_label'] = sentiment['label']
                
                flat_data.append(row)
            
            df = pd.DataFrame(flat_data)
            path = os.path.join(output_dir, f"article_sentiment_{timestamp}.csv")
            df.to_csv(path, index=False, encoding='utf-8')
        
        # Sentiment par topic
        if self.results['topic_sentiment']:
            flat_data = []
            for topic, data in self.results['topic_sentiment'].items():
                for model_name, sentiment in data['aggregated_sentiment'].items():
                    row = {
                        'topic': topic,
                        'model': model_name,
                        'total_articles': data['total_articles']
                    }
                    
                    if model_name == 'textblob':
                        row['avg_polarity'] = sentiment['avg_polarity']
                        row['avg_subjectivity'] = sentiment['avg_subjectivity']
                    elif model_name == 'vader':
                        row['avg_compound'] = sentiment['avg_compound']
                        row['avg_positive'] = sentiment['avg_positive']
                        row['avg_negative'] = sentiment['avg_negative']
                        row['avg_neutral'] = sentiment['avg_neutral']
                    
                    # Distribution
                    for label, count in sentiment['distribution'].items():
                        row[f'{label}_count'] = count
                    
                    flat_data.append(row)
            
            if flat_data:
                df = pd.DataFrame(flat_data)
                path = os.path.join(output_dir, f"topic_sentiment_{timestamp}.csv")
                df.to_csv(path, index=False, encoding='utf-8')


def main():
    """Fonction principale de test"""
    
    # Trouver le corpus prétraité
    import glob
    preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
    
    if not preprocessed_files:
        print("Aucun corpus prétraité trouvé. Lancez d'abord le prétraitement.")
        return
    
    corpus_path = max(preprocessed_files)
    print(f"Analyse de sentiment du corpus: {corpus_path}")
    
    # Initialiser l'analyseur
    analyzer = SentimentAnalyzer()
    analyzer.load_corpus(corpus_path)
    
    # Effectuer les analyses
    results = analyzer.analyze_all()
    
    # Afficher un résumé
    print("\n" + "="*60)
    print("RÉSUMÉ DE L'ANALYSE DE SENTIMENT APPROFONDIE")
    print("="*60)
    
    # Sentiment par topic
    print("\n📊 SENTIMENT PAR TOPIC:")
    for topic, data in results['topic_sentiment'].items():
        print(f"\n{topic.upper()}:")
        print(f"  • Articles: {data['total_articles']}")
        print(f"  • Consensus: {data['consensus_distribution']}")
        print(f"  • Ratio: Positif={data['sentiment_ratio']['positive']:.1%}, "
              f"Négatif={data['sentiment_ratio']['negative']:.1%}, "
              f"Neutre={data['sentiment_ratio']['neutral']:.1%}")
    
    # Acteurs analysés
    print("\n🎯 ACTEURS ANALYSÉS:")
    for topic in results.get('target_word_sentiment', {}):
        actors = list(results['target_word_sentiment'][topic].keys())
        print(f"\n{topic.upper()} ({len(actors)} acteurs):")
        # Afficher par groupes de 5
        for i in range(0, len(actors), 5):
            print(f"  {', '.join(actors[i:i+5])}")
    
    # Comparaisons
    print("\n⚖️ COMPARAISONS ENTRE TOPICS:")
    for comparison, data in results.get('sentiment_comparison', {}).items():
        print(f"\n{comparison}:")
        for label, diff in list(data.items())[:3]:  # Afficher seulement 3
            diff_val = diff['difference']
            if abs(diff_val) > 10:
                direction = "plus positif pour" if diff_val < 0 else "plus négatif pour"
                topic = comparison.split('_vs_')[1 if diff_val < 0 else 0]
                print(f"  • {label}: {abs(diff_val):.1f}% {direction} {topic}")
    
    # Analyse comparative des acteurs
    print("\n🔍 COMPARAISON DES ACTEURS SIMILAIRES:")
    if 'actor_comparative_analysis' in results:
        for actor_type, comparison in results['actor_comparative_analysis'].items():
            if 'comparison' in comparison:
                print(f"\n{actor_type.upper()}:")
                for sentiment, stats in comparison['comparison'].items():
                    if abs(stats['difference']) > 15:  # Seulement les différences significatives
                        direction = "→ Gaza" if stats['difference'] > 0 else "→ Ukraine"
                        print(f"  • {sentiment}: {abs(stats['difference']):.1f}% {direction}")


if __name__ == "__main__":
    main()