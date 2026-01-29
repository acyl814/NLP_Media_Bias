"""
Collecteurs de données pour les articles de presse
Supporte CNN, BBC, New York Times
"""

import os
import time
import json
import requests
from datetime import datetime
from urllib.parse import urljoin, quote
from bs4 import BeautifulSoup
from newspaper import Article
import pandas as pd
import yaml
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsCollector:
    """Collecteur générique pour les articles de presse"""
    
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.sources = {s['name']: s for s in self.config['data_collection']['sources']}
        self.keywords = self.config['data_collection']['keywords']
        self.limits = self.config['data_collection']['limits']
        
        # Headers pour éviter les blocages
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        
        # Session pour maintenir les cookies
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def collect_all(self, output_dir="corpus"):
        """Collecte tous les articles pour Gaza et Ukraine"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_articles = []
        
        # Collecte pour Gaza
        logger.info("Collecte des articles sur Gaza...")
        gaza_articles = self.collect_by_topic("gaza", self.limits['gaza'])
        for article in gaza_articles:
            article['topic'] = 'gaza'
        all_articles.extend(gaza_articles)
        
        # Collecte pour Ukraine
        logger.info("Collecte des articles sur l'Ukraine...")
        ukraine_articles = self.collect_by_topic("ukraine", self.limits['ukraine'])
        for article in ukraine_articles:
            article['topic'] = 'ukraine'
        all_articles.extend(ukraine_articles)
        
        # Sauvegarde
        self.save_articles(all_articles, output_dir)
        
        logger.info(f"Collecte terminée: {len(all_articles)} articles collectés")
        return all_articles
    
    def collect_by_topic(self, topic, limit):
        """Collecte les articles pour un sujet spécifique"""
        
        articles = []
        keywords = self.keywords[topic]
        
        for source_name in self.sources.keys():
            if len(articles) >= limit:
                break
                
            logger.info(f"Collecte depuis {source_name} pour {topic}...")
            source_articles = self.collect_from_source(source_name, topic, keywords, limit - len(articles))
            articles.extend(source_articles)
        
        return articles[:limit]
    
    def collect_from_source(self, source_name, topic, keywords, limit):
        """Collecte les articles d'une source spécifique"""
        
        source_config = self.sources[source_name]
        articles = []
        
        for keyword in keywords:
            if len(articles) >= limit:
                break
                
            try:
                # Recherche d'articles
                search_urls = self._generate_search_urls(source_name, keyword)
                
                for search_url in search_urls:
                    if len(articles) >= limit:
                        break
                    
                    logger.info(f"Recherche: {keyword} sur {source_name}")
                    article_urls = self._extract_article_urls(source_name, search_url)
                    
                    for url in article_urls[:limit - len(articles)]:
                        try:
                            article = self._extract_article_content(url)
                            if article and self._is_relevant(article, topic):
                                article['source'] = source_name
                                article['keyword'] = keyword
                                articles.append(article)
                                
                                # Délai entre les requêtes
                                time.sleep(self.limits['requests_delay'])
                                
                        except Exception as e:
                            logger.warning(f"Erreur lors de l'extraction de {url}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Erreur lors de la collecte depuis {source_name} pour {keyword}: {e}")
                continue
        
        return articles
    
    def _generate_search_urls(self, source_name, keyword):
        """Génère les URLs de recherche"""
        
        source = self.sources[source_name]
        
        if source_name == "cnn":
            # CNN utilise une API de recherche
            return [f"https://edition.cnn.com/search?q={quote(keyword)}&size=50"]
        
        elif source_name == "bbc":
            # BBC recherche
            return [f"https://www.bbc.co.uk/search?q={quote(keyword)}&filter=news"]
        
        elif source_name == "nytimes":
            # NYTimes recherche
            return [f"https://www.nytimes.com/search?query={quote(keyword)}"]
        
        return []
    
    def _extract_article_urls(self, source_name, search_url):
        """Extrait les URLs des articles depuis une page de recherche"""
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            source = self.sources[source_name]
            
            # Extraction selon la source
            if source_name == "cnn":
                links = soup.select("a.container__link")
                base_url = "https://edition.cnn.com"
            
            elif source_name == "bbc":
                links = soup.select("a.ssrcss-1mrs5ns-LinkPostLink")
                base_url = "https://www.bbc.com"
            
            elif source_name == "nytimes":
                links = soup.select("a.css-1mshvvr")
                base_url = "https://www.nytimes.com"
            
            else:
                return []
            
            # Nettoyage et filtrage des URLs
            urls = []
            for link in links:
                href = link.get('href', '')
                if href and href.startswith('/'):
                    full_url = urljoin(base_url, href)
                    if self._is_valid_article_url(full_url, source_name):
                        urls.append(full_url)
            
            return list(set(urls[:20]))  # Limiter et dédupliquer
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction des URLs depuis {search_url}: {e}")
            return []
    
    def _extract_article_content(self, url):
        """Extrait le contenu d'un article avec newspaper3k"""
        
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()
            
            if not article.title or not article.text:
                return None
            
            # Extraction manuelle si newspaper échoue
            if len(article.text) < 100:
                article = self._manual_extraction(url)
                if not article:
                    return None
            
            return {
                'url': url,
                'title': article.title,
                'content': article.text,
                'authors': article.authors if hasattr(article, 'authors') else [],
                'publish_date': article.publish_date.isoformat() if article.publish_date else None,
                'extraction_date': datetime.now().isoformat(),
                'word_count': len(article.text.split())
            }
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction de {url}: {e}")
            return None
    
    def _manual_extraction(self, url):
        """Extraction manuelle si newspaper3k échoue"""
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Supprimer les scripts et styles
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Trouver le contenu principal
            content_selectors = [
                'article',
                '[class*="article"]',
                '[class*="content"]',
                '[class*="story"]',
                'main'
            ]
            
            content = ""
            for selector in content_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(separator=' ', strip=True)
                    if len(text) > len(content):
                        content = text
            
            # Si toujours rien, prendre tout le body
            if not content:
                body = soup.find('body')
                if body:
                    content = body.get_text(separator=' ', strip=True)
            
            # Nettoyer le texte
            content = ' '.join(content.split())
            
            # Titre
            title = ""
            title_selectors = ['h1', '[class*="title"]', '[class*="headline"]']
            for selector in title_selectors:
                element = soup.select_one(selector)
                if element:
                    title = element.get_text(strip=True)
                    break
            
            if content and len(content) > 100:
                return type('obj', (object,), {
                    'title': title or url,
                    'text': content,
                    'authors': [],
                    'publish_date': None
                })()
            
            return None
            
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction manuelle de {url}: {e}")
            return None
    
    def _is_valid_article_url(self, url, source_name):
        """Vérifie si l'URL pointe vers un article valide"""
        
        if not url or not url.startswith('http'):
            return False
        
        # Filtres par source
        if source_name == "cnn":
            return '/2024/' in url or '/2023/' in url or '/2025/' in url
        
        elif source_name == "bbc":
            return '/news/' in url and ('-6' in url or '-7' in url)  # IDs d'articles
        
        elif source_name == "nytimes":
            return '/2024/' in url or '/2023/' in url or '/2025/' in url
        
        return False
    
    def _is_relevant(self, article, topic):
        """Vérifie si l'article est pertinent pour le sujet"""
        
        text = f"{article['title']} {article['content']}".lower()
        keywords = self.keywords[topic]
        
        # Au moins 2 mots-clés doivent être présents
        matches = sum(1 for kw in keywords if kw.lower() in text)
        return matches >= 2
    
    def save_articles(self, articles, output_dir):
        """Sauvegarde les articles au format JSON et CSV"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = os.path.join(output_dir, f"corpus_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        # CSV
        df = pd.DataFrame(articles)
        csv_path = os.path.join(output_dir, f"corpus_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Statistiques
        stats = {
            'total_articles': len(articles),
            'topics': df['topic'].value_counts().to_dict(),
            'sources': df['source'].value_counts().to_dict(),
            'total_words': df['word_count'].sum(),
            'avg_words': df['word_count'].mean()
        }
        
        stats_path = os.path.join(output_dir, f"stats_{timestamp}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Articles sauvegardés: {json_path}, {csv_path}")
        logger.info(f"Statistiques: {stats}")


def main():
    """Fonction principale"""
    
    collector = NewsCollector()
    articles = collector.collect_all("corpus")
    
    print(f"\n{'='*60}")
    print("COLLECTE TERMINÉE")
    print(f"{'='*60}")
    print(f"Total d'articles collectés: {len(articles)}")
    
    # Répartition par sujet
    topics = {}
    for article in articles:
        topic = article.get('topic', 'unknown')
        topics[topic] = topics.get(topic, 0) + 1
    
    print("\nRépartition par sujet:")
    for topic, count in topics.items():
        print(f"  - {topic}: {count} articles")
    
    # Répartition par source
    sources = {}
    for article in articles:
        source = article.get('source', 'unknown')
        sources[source] = sources.get(source, 0) + 1
    
    print("\nRépartition par source:")
    for source, count in sources.items():
        print(f"  - {source}: {count} articles")


if __name__ == "__main__":
    main()