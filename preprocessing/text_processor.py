"""
Module de prétraitement des textes pour l'analyse NLP
Supporte le nettoyage, la lemmatisation et la préparation des données
"""
# Ajouter en haut du fichier (vers la ligne 1-5) :
from datetime import datetime  # <-- IMPORTANT !
import json
import os
import re
# ... autres importscle
import re
import string
import json
import os
from typing import List, Dict, Tuple, Optional

import nltk
import spacy
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import yaml
from tqdm import tqdm
import logging

# Télécharger les ressources nécessaires de NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextProcessor:
    """Processeur de textes pour le nettoyage et le prétraitement NLP"""
    
    def __init__(self, config_path="config/config.yaml", use_spacy=True):
        """
        Initialise le processeur de textes
        
        Args:
            config_path: Chemin vers le fichier de configuration
            use_spacy: Utiliser spaCy pour le traitement (plus lent mais plus précis)
        """
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessing_config = self.config['preprocessing']
        self.use_spacy = use_spacy
        
        # Initialiser spaCy si disponible
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy chargé avec succès")
            except OSError:
                logger.warning("spaCy non trouvé, installation en cours...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
        
        # Initialiser NLTK
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Ajouter des mots d'arrêt spécifiques au domaine
        domain_stopwords = {
            'said', 'say', 'says', 'told', 'according', 'reported', 'also',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'one', 'two', 'three', 'first', 'second', 'last', 'new', 'old',
            'year', 'years', 'day', 'days', 'time', 'times', 'people',
            'gaza', 'ukraine', 'israel', 'russia', 'palestinian', 'ukrainian'
        }
        self.stop_words.update(domain_stopwords)
        
        # Patterns regex pour le nettoyage
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\S+@\S+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.mention_pattern = re.compile(r'@\w+')
        self.extra_whitespace = re.compile(r'\s+')
        
        # Patterns pour les entités à garder
        self.person_names = set()
        self.location_names = set()
    
    def process_corpus(self, corpus_path: str, output_dir: str = "preprocessed") -> str:
        """
        Traite un corpus complet
        
        Args:
            corpus_path: Chemin vers le fichier corpus JSON ou CSV
            output_dir: Répertoire de sortie
            
        Returns:
            Chemin vers le fichier prétraité
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Charger le corpus
        if corpus_path.endswith('.json'):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus = json.load(f)
        elif corpus_path.endswith('.csv'):
            df = pd.read_csv(corpus_path)
            corpus = df.to_dict('records')
        else:
            raise ValueError("Format de fichier non supporté. Utilisez JSON ou CSV.")
        
        logger.info(f"Traitement de {len(corpus)} articles...")
        
        # Traiter chaque article
        processed_articles = []
        for article in tqdm(corpus, desc="Processing articles"):
            processed = self.process_article(article)
            processed_articles.append(processed)
        
        # Sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"preprocessed_{timestamp}.json")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, indent=2, ensure_ascii=False)
        
        # Créer un fichier de texte concaténé pour l'analyse
        self._create_combined_text(processed_articles, output_dir, timestamp)
        
        logger.info(f"Corpus prétraité sauvegardé dans: {output_path}")
        return output_path
    
    def process_article(self, article: Dict) -> Dict:
        """
        Traite un article unique
        
        Args:
            article: Dictionnaire contenant les données de l'article
            
        Returns:
            Article avec texte prétraité
        """
        
        # Traiter le titre
        title_clean = self.clean_text(article['title'])
        title_tokens = self.tokenize(title_clean)
        
        # Traiter le contenu
        content_clean = self.clean_text(article['content'])
        content_tokens = self.tokenize(content_clean)
        
        # Extraire les phrases
        sentences = self.extract_sentences(content_clean)
        
        # Extraire les entités nommées
        entities = self.extract_entities(content_clean)
        
        # Calculer les statistiques
        stats = {
            'original_word_count': len(article['content'].split()),
            'cleaned_word_count': len(content_tokens),
            'sentence_count': len(sentences),
            'unique_words': len(set(content_tokens)),
            'lexical_diversity': len(set(content_tokens)) / len(content_tokens) if content_tokens else 0
        }
        
        # Créer l'article prétraité
        processed = article.copy()
        processed.update({
            'title_clean': title_clean,
            'title_tokens': title_tokens,
            'content_clean': content_clean,
            'content_tokens': content_tokens,
            'sentences': sentences,
            'entities': entities,
            'processing_stats': stats
        })
        
        return processed
    
    def clean_text(self, text: str) -> str:
        """
        Nettoie le texte brut
        
        Args:
            text: Texte brut
            
        Returns:
            Texte nettoyé
        """
        
        if not text or not isinstance(text, str):
            return ""
        
        # Supprimer les URLs
        text = self.url_pattern.sub('', text)
        
        # Supprimer les emails
        text = self.email_pattern.sub('', text)
        
        # Supprimer les hashtags et mentions (pour les réseaux sociaux)
        text = self.hashtag_pattern.sub('', text)
        text = self.mention_pattern.sub('', text)
        
        # Supprimer les numéros si configuré
        if self.preprocessing_config.get('remove_numbers', False):
            text = re.sub(r'\d+', '', text)
        
        # Supprimer la ponctuation si configuré
        if self.preprocessing_config.get('remove_punctuation', False):
            # Garder les apostrophes pour ne pas casser les mots
            text = text.translate(str.maketrans('', '', string.punctuation.replace("'", "")))
        
        # Mettre en minuscule
        if self.preprocessing_config.get('lowercase', False):
            text = text.lower()
        
        # Nettoyer les espaces
        text = self.extra_whitespace.sub(' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenise le texte et applique lemmatization
        
        Args:
            text: Texte à tokeniser
            
        Returns:
            Liste de tokens
        """
        
        if not text:
            return []
        
        if self.use_spacy and hasattr(self, 'nlp'):
            # Utiliser spaCy
            doc = self.nlp(text)
            tokens = []
            
            for token in doc:
                # Filtrer les tokens
                if self._is_valid_token(token):
                    if self.preprocessing_config.get('lemmatization', False):
                        tokens.append(token.lemma_.lower())
                    else:
                        tokens.append(token.text.lower())
            
            return tokens
        
        else:
            # Utiliser NLTK
            tokens = word_tokenize(text)
            
            # Filtrer et lemmatiser
            filtered_tokens = []
            for token in tokens:
                if self._is_valid_token_nltk(token):
                    if self.preprocessing_config.get('lemmatization', False):
                        filtered_tokens.append(self.lemmatizer.lemmatize(token.lower()))
                    else:
                        filtered_tokens.append(token.lower())
            
            return filtered_tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extrait les phrases du texte
        
        Args:
            text: Texte à segmenter
            
        Returns:
            Liste de phrases
        """
        
        if self.use_spacy and hasattr(self, 'nlp'):
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            sentences = sent_tokenize(text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les entités nommées du texte
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des entités par type
        """
        
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],  # Countries, cities, states
            'NORP': [],  # Nationalities or religious or political groups
            'EVENT': [],
            'DATE': []
        }
        
        if self.use_spacy and hasattr(self, 'nlp'):
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
                
                # Collecte des noms pour référence
                if ent.label_ == 'PERSON':
                    self.person_names.add(ent.text)
                elif ent.label_ in ['GPE', 'NORP']:
                    self.location_names.add(ent.text)
        
        # Nettoyer les doublons
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities
    
    def _is_valid_token(self, token) -> bool:
        """Vérifie si un token spaCy est valide"""
        
        # Pas de ponctuation
        if token.is_punct:
            return False
        
        # Pas d'espaces
        if token.is_space:
            return False
        
        # Longueur minimale
        min_length = self.preprocessing_config.get('min_word_length', 3)
        if len(token.text) < min_length:
            return False
        
        # Pas de stop words
        if self.preprocessing_config.get('remove_stopwords', False):
            if token.text.lower() in self.stop_words:
                return False
        
        return True
    
    def _is_valid_token_nltk(self, token: str) -> bool:
        """Vérifie si un token NLTK est valide"""
        
        # Pas de ponctuation pure
        if token in string.punctuation:
            return False
        
        # Longueur minimale
        min_length = self.preprocessing_config.get('min_word_length', 3)
        if len(token) < min_length:
            return False
        
        # Pas de stop words
        if self.preprocessing_config.get('remove_stopwords', False):
            if token.lower() in self.stop_words:
                return False
        
        return True
    
    def _create_combined_text(self, articles: List[Dict], output_dir: str, timestamp: str):
        """
        Crée un fichier texte concaténé pour l'analyse
        
        Args:
            articles: Liste d'articles prétraités
            output_dir: Répertoire de sortie
            timestamp: Timestamp pour le nom de fichier
        """
        
        # Par topic
        topics = {}
        for article in articles:
            topic = article.get('topic', 'unknown')
            if topic not in topics:
                topics[topic] = []
            topics[topic].append(article['content_clean'])
        
        # Sauvegarder les fichiers combinés
        for topic, texts in topics.items():
            combined_path = os.path.join(output_dir, f"{topic}_combined_{timestamp}.txt")
            with open(combined_path, 'w', encoding='utf-8') as f:
                f.write('\n\n'.join(texts))
        
        # Fichier complet
        all_texts = [article['content_clean'] for article in articles]
        combined_path = os.path.join(output_dir, f"all_combined_{timestamp}.txt")
        with open(combined_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(all_texts))
    
    def get_vocabulary_stats(self, articles: List[Dict]) -> Dict:
        """
        Calcule les statistiques de vocabulaire
        
        Args:
            articles: Liste d'articles prétraités
            
        Returns:
            Statistiques de vocabulaire
        """
        
        all_tokens = []
        topic_tokens = {}
        
        for article in articles:
            topic = article.get('topic', 'unknown')
            tokens = article.get('content_tokens', [])
            
            all_tokens.extend(tokens)
            
            if topic not in topic_tokens:
                topic_tokens[topic] = []
            topic_tokens[topic].extend(tokens)
        
        stats = {
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'lexical_diversity': len(set(all_tokens)) / len(all_tokens) if all_tokens else 0,
            'topics': {}
        }
        
        for topic, tokens in topic_tokens.items():
            stats['topics'][topic] = {
                'total_tokens': len(tokens),
                'unique_tokens': len(set(tokens)),
                'lexical_diversity': len(set(tokens)) / len(tokens) if tokens else 0
            }
        
        return stats


def main():
    """Fonction principale de test"""
    
    # Créer un exemple de corpus si nécessaire
    if not os.path.exists("corpus"):
        print("Aucun corpus trouvé. Lancez d'abord le générateur d'exemples.")
        return
    
    # Trouver le dernier fichier corpus
    import glob
    corpus_files = glob.glob("corpus/corpus_*.json")
    if not corpus_files:
        print("Aucun fichier corpus trouvé.")
        return
    
    corpus_path = max(corpus_files)  # Le plus récent
    print(f"Traitement du corpus: {corpus_path}")
    
    # Initialiser le processeur
    processor = TextProcessor(use_spacy=False)  # NLTK plus rapide pour les tests
    
    # Traiter le corpus
    output_path = processor.process_corpus(corpus_path)
    
    # Afficher les statistiques
    with open(output_path, 'r', encoding='utf-8') as f:
        processed_articles = json.load(f)
    
    stats = processor.get_vocabulary_stats(processed_articles)
    
    print("\n" + "="*60)
    print("STATISTIQUES DE VOCABULAIRE")
    print("="*60)
    print(f"Total de tokens: {stats['total_tokens']:,}")
    print(f"Tokens uniques: {stats['unique_tokens']:,}")
    print(f"Diversité lexicale: {stats['lexical_diversity']:.3f}")
    
    print("\nPar sujet:")
    for topic, topic_stats in stats['topics'].items():
        print(f"\n  {topic.upper()}:")
        print(f"    Tokens: {topic_stats['total_tokens']:,}")
        print(f"    Uniques: {topic_stats['unique_tokens']:,}")
        print(f"    Diversité: {topic_stats['lexical_diversity']:.3f}")


if __name__ == "__main__":
    from datetime import datetime
    main()