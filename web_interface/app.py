"""
Application Web Flask pour l'interface utilisateur
Permet d'explorer les corpus et visualiser les résultats
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_cors import CORS
import pandas as pd
import yaml
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialiser Flask
app = Flask(
    __name__,
    static_folder="../visualizations",
    static_url_path="/viz"
)
CORS(app)

# Configuration
app.config['SECRET_KEY'] = 'votre-secret-key-ici'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Variables globales
CONFIG = {}
CORPUS_DATA = []
ANALYSIS_RESULTS = {}


def load_config():
    """Charge la configuration"""
    global CONFIG
    
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
        logger.info("Configuration chargée")
    else:
        logger.warning("Fichier de configuration non trouvé")


def load_corpus_data():
    """Charge les données du corpus"""
    global CORPUS_DATA
    
    # Trouver le dernier corpus
    corpus_files = glob.glob("corpus/corpus_*.json")
    
    if corpus_files:
        latest_corpus = max(corpus_files)
        
        with open(latest_corpus, 'r', encoding='utf-8') as f:
            CORPUS_DATA = json.load(f)
        
        logger.info(f"Corpus chargé: {len(CORPUS_DATA)} articles")
    else:
        logger.warning("Aucun fichier corpus trouvé")


def load_analysis_results():
    """Charge les résultats d'analyse"""
    global ANALYSIS_RESULTS
    
    # Initialiser la structure des résultats
    ANALYSIS_RESULTS = {
        "lexical": {},
        "semantic": {},
        "sentiment": {},
        "comparison": {}
    }
    
    results_dir = "analysis_results"
    if not os.path.exists(results_dir):
        logger.warning(f"Le dossier {results_dir} n'existe pas")
        return
    
    # Charger les analyses lexicales
    lexical_files = glob.glob(f"{results_dir}/lexical_analysis_*.json")
    if lexical_files:
        try:
            latest_lexical = max(lexical_files)
            with open(latest_lexical, 'r', encoding='utf-8') as f:
                ANALYSIS_RESULTS['lexical'] = json.load(f)
            logger.info(f"Résultats lexicaux chargés: {latest_lexical}")
        except Exception as e:
            logger.error(f"Erreur chargement lexical: {e}")
    
    # Charger les analyses sémantiques
    semantic_files = glob.glob(f"{results_dir}/semantic_analysis_*.json")
    if semantic_files:
        try:
            latest_semantic = max(semantic_files)
            with open(latest_semantic, 'r', encoding='utf-8') as f:
                ANALYSIS_RESULTS['semantic'] = json.load(f)
            logger.info(f"Résultats sémantiques chargés: {latest_semantic}")
        except Exception as e:
            logger.error(f"Erreur chargement sémantique: {e}")
    
    # Charger les analyses de sentiment
    sentiment_files = glob.glob(f"{results_dir}/sentiment_analysis_*.json")
    if sentiment_files:
        try:
            latest_sentiment = max(sentiment_files)
            with open(latest_sentiment, 'r', encoding='utf-8') as f:
                ANALYSIS_RESULTS['sentiment'] = json.load(f)
            logger.info(f"Résultats sentiment chargés: {latest_sentiment}")
        except Exception as e:
            logger.error(f"Erreur chargement sentiment: {e}")
    
    # Charger les analyses de comparaison (si existent)
    comparison_files = glob.glob(f"{results_dir}/comparison_*.json")
    if comparison_files:
        try:
            latest_comparison = max(comparison_files)
            with open(latest_comparison, 'r', encoding='utf-8') as f:
                ANALYSIS_RESULTS['comparison'] = json.load(f)
            logger.info(f"Résultats comparaison chargés: {latest_comparison}")
        except Exception as e:
            logger.error(f"Erreur chargement comparaison: {e}")


# Routes de l'application
@app.route('/')
def index():
    """Page d'accueil"""
    
    stats = {
        'total_articles': len(CORPUS_DATA),
        'total_words': sum(len(article.get('content', '').split()) for article in CORPUS_DATA),
        'topics': {}
    }
    
    # Statistiques par topic
    for article in CORPUS_DATA:
        topic = article.get('topic', 'unknown')
        if topic not in stats['topics']:
            stats['topics'][topic] = 0
        stats['topics'][topic] += 1
    
    return render_template('index.html', 
                         stats=stats,
                         config=CONFIG)


@app.route('/corpus')
def corpus_explorer():
    """Explorateur de corpus"""
    
    # Paramètres de pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    topic_filter = request.args.get('topic', 'all')
    source_filter = request.args.get('source', 'all')
    
    # Filtrer les articles
    filtered_articles = CORPUS_DATA.copy()
    
    if topic_filter != 'all':
        filtered_articles = [art for art in filtered_articles if art.get('topic') == topic_filter]
    
    if source_filter != 'all':
        filtered_articles = [art for art in filtered_articles if art.get('source') == source_filter]
    
    # Pagination
    total_articles = len(filtered_articles)
    total_pages = (total_articles + per_page - 1) // per_page
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    
    articles = filtered_articles[start_idx:end_idx]
    
    # Liste des topics et sources pour les filtres
    topics = list(set(art.get('topic', 'unknown') for art in CORPUS_DATA))
    sources = list(set(art.get('source', 'unknown') for art in CORPUS_DATA))
    
    return render_template('corpus.html',
                         articles=articles,
                         page=page,
                         total_pages=total_pages,
                         topics=topics,
                         sources=sources,
                         topic_filter=topic_filter,
                         source_filter=source_filter)


@app.route('/analysis')
def analysis_dashboard():
    # ... chargement des données ...
    
    # Nettoyer les données sémantiques problématiques
    if 'semantic' in ANALYSIS_RESULTS:
        semantic_data = ANALYSIS_RESULTS['semantic'].copy()
        
        # Supprimer les données corrompues
        for key in ['word2vec', 'actor_semantics', 'actors']:
            if key in semantic_data:
                # Vérifier si les données sont valides
                try:
                    # Test simple: vérifier le premier élément
                    test_val = list(semantic_data[key].values())[0]
                    if isinstance(test_val, dict):
                        first_subval = list(test_val.values())[0]
                        if isinstance(first_subval, list) and len(first_subval) > 0:
                            first_item = first_subval[0]
                            if isinstance(first_item, str) and "\\'" in first_item:
                                # Données corrompues, les supprimer
                                del semantic_data[key]
                except:
                    del semantic_data[key]
        
        ANALYSIS_RESULTS['semantic'] = semantic_data
    
    return render_template('analysis.html',
                         results=ANALYSIS_RESULTS,
                         config=CONFIG)

@app.route('/regenerate-semantic')
def regenerate_semantic():
    """Route pour régénérer l'analyse sémantique"""
    try:
        # Importer et exécuter le régénérateur
        from analysis.regenerate_semantic import SemanticRegenerator
        
        regenerator = SemanticRegenerator()
        results = regenerator.run_complete_analysis()
        
        # Mettre à jour les résultats en mémoire
        global ANALYSIS_RESULTS
        ANALYSIS_RESULTS['semantic'] = results
        
        flash('✅ Analyse sémantique régénérée avec succès!', 'success')
        
    except Exception as e:
        app.logger.error(f"Erreur régénération sémantique: {e}", exc_info=True)
        flash(f'❌ Erreur lors de la régénération: {str(e)}', 'danger')
    
    return redirect(url_for('analysis_dashboard'))

@app.route('/bias-detector')
def bias_detector():
    """Détecteur de biais interactif"""
    
    return render_template('bias_detector.html',
                         results=ANALYSIS_RESULTS,
                         config=CONFIG)


@app.route('/visualizations')
def visualizations():
    """Galerie de visualisations"""
    
    # Liste des visualisations disponibles
    viz_dir = "visualizations"
    visualizations_list = []
    
    if os.path.exists(viz_dir):
        for file in os.listdir(viz_dir):
            if file.endswith('.html'):
                visualizations_list.append({
                    'name': file.replace('.html', '').replace('_', ' ').title(),
                    'file': file,
                    'type': 'interactive'
                })
            elif file.endswith('.png'):
                visualizations_list.append({
                    'name': file.replace('.png', '').replace('_', ' ').title(),
                    'file': file,
                    'type': 'image'
                })
    
    return render_template('visualizations.html',
                         visualizations=visualizations_list)


@app.route('/api/articles')
def api_articles():
    """API pour récupérer les articles"""
    
    topic = request.args.get('topic')
    source = request.args.get('source')
    limit = request.args.get('limit', default=50, type=int)
    
    articles = CORPUS_DATA.copy()
    
    if topic:
        articles = [art for art in articles if art.get('topic') == topic]
    
    if source:
        articles = [art for art in articles if art.get('source') == source]
    
    # Limiter le nombre de résultats
    articles = articles[:limit]
    
    return jsonify({
        'articles': articles,
        'total': len(articles)
    })


@app.route('/api/analysis/<analysis_type>')
def api_analysis(analysis_type):
    """API pour récupérer les résultats d'analyse"""
    
    if analysis_type in ANALYSIS_RESULTS:
        return jsonify(ANALYSIS_RESULTS[analysis_type])
    else:
        return jsonify({'error': 'Type d\'analyse non trouvé'}), 404


@app.route('/api/stats')
def api_stats():
    """API pour récupérer les statistiques"""
    
    stats = {
        'total_articles': len(CORPUS_DATA),
        'total_words': sum(len(article.get('content', '').split()) for article in CORPUS_DATA),
        'topics': {},
        'sources': {},
        'date_range': {
            'earliest': None,
            'latest': None
        }
    }
    
    # Statistiques par topic
    for article in CORPUS_DATA:
        topic = article.get('topic', 'unknown')
        source = article.get('source', 'unknown')
        
        stats['topics'][topic] = stats['topics'].get(topic, 0) + 1
        stats['sources'][source] = stats['sources'].get(source, 0) + 1
        
        # Dates
        publish_date = article.get('publish_date')
        if publish_date:
            try:
                date_obj = pd.to_datetime(publish_date)
                if stats['date_range']['earliest'] is None or date_obj < pd.to_datetime(stats['date_range']['earliest']):
                    stats['date_range']['earliest'] = publish_date
                if stats['date_range']['latest'] is None or date_obj > pd.to_datetime(stats['date_range']['latest']):
                    stats['date_range']['latest'] = publish_date
            except:
                pass
    
    return jsonify(stats)


@app.route('/api/search')
def api_search():
    """API pour rechercher dans les articles"""
    
    query = request.args.get('q', '').lower()
    topic = request.args.get('topic')
    
    results = []
    
    for article in CORPUS_DATA:
        # Filtrer par topic si spécifié
        if topic and article.get('topic') != topic:
            continue
        
        # Rechercher dans le titre et le contenu
        title = article.get('title', '').lower()
        content = article.get('content', '').lower()
        
        if query in title or query in content:
            # Calculer un score de pertinence
            title_score = title.count(query) * 3  # Plus de poids pour le titre
            content_score = content.count(query)
            relevance_score = title_score + content_score
            
            result = article.copy()
            result['relevance_score'] = relevance_score
            results.append(result)
    
    # Trier par score de pertinence
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return jsonify({
        'query': query,
        'results': results[:20],  # Limiter les résultats
        'total': len(results)
    })


@app.route('/dashboard')
def dashboard():
    """Tableau de bord unifié"""
    
    # Calculer des statistiques rapides
    stats = {
        'total_articles': len(CORPUS_DATA),
        'topics': {},
        'analyses_available': {}
    }
    
    for article in CORPUS_DATA:
        topic = article.get('topic', 'unknown')
        stats['topics'][topic] = stats['topics'].get(topic, 0) + 1
    
    # Vérifier quelles analyses sont disponibles
    if ANALYSIS_RESULTS.get('lexical'):
        stats['analyses_available']['lexical'] = True
    if ANALYSIS_RESULTS.get('semantic'):
        stats['analyses_available']['semantic'] = True
    if ANALYSIS_RESULTS.get('sentiment'):
        stats['analyses_available']['sentiment'] = True
    
    return render_template('dashboard.html', 
                         results=ANALYSIS_RESULTS,
                         stats=stats,
                         config=CONFIG)


@app.route('/report')
def generate_report():
    """Génère un rapport HTML"""
    
    # Récupérer les données pour le rapport
    corpus_stats = {
        'total_articles': len(CORPUS_DATA),
        'total_words': sum(len(article.get('content', '').split()) for article in CORPUS_DATA),
        'topics': {}
    }
    
    for article in CORPUS_DATA:
        topic = article.get('topic', 'unknown')
        corpus_stats['topics'][topic] = corpus_stats['topics'].get(topic, 0) + 1
    
    return render_template('report.html',
                         results=ANALYSIS_RESULTS,
                         corpus_stats=corpus_stats,
                         config=CONFIG,
                         generation_date=datetime.now().strftime("%d/%m/%Y %H:%M"))


@app.route('/run-analysis/lexical')
def run_lexical_analysis():
    """Lance l'analyse lexicale"""
    try:
        # Trouver le dernier corpus prétraité
        preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
        if not preprocessed_files:
            return jsonify({
                "success": False,
                "message": "Aucun corpus prétraité trouvé"
            })
        
        corpus_path = max(preprocessed_files)
        
        # Importer et exécuter l'analyseur lexical
        from analysis.lexical_analyzer import LexicalAnalyzer
        
        analyzer = LexicalAnalyzer()
        analyzer.load_corpus(corpus_path)
        results = analyzer.analyze_all()
        
        # Recharger les résultats dans l'interface
        global ANALYSIS_RESULTS
        ANALYSIS_RESULTS["lexical"] = results
        
        # Recharger tous les résultats pour synchronisation
        load_analysis_results()
        
        return jsonify({
            "success": True,
            "message": f"Analyse lexicale terminée.",
            "redirect": "/analysis#lexical"
        })
        
    except Exception as e:
        logger.error(f"Erreur analyse lexicale: {e}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        })


@app.route('/run-analysis/semantic')
def run_semantic_analysis():
    """Lance l'analyse sémantique"""
    try:
        # Trouver le dernier corpus prétraité
        preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
        if not preprocessed_files:
            return jsonify({
                "success": False,
                "message": "Aucun corpus prétraité trouvé"
            })
        
        corpus_path = max(preprocessed_files)
        
        # Importer et exécuter l'analyseur sémantique
        from analysis.semantic_analyzer import EnhancedSemanticAnalyzer
        
        analyzer = EnhancedSemanticAnalyzer()
        analyzer.load_corpus(corpus_path)
        results = analyzer.analyze_all()
        
        # Recharger les résultats dans l'interface
        global ANALYSIS_RESULTS
        ANALYSIS_RESULTS["semantic"] = results
        
        # Recharger tous les résultats pour synchronisation
        load_analysis_results()
        
        return jsonify({
            "success": True,
            "message": f"Analyse sémantique terminée.",
            "redirect": "/analysis#semantic"
        })
        
    except Exception as e:
        logger.error(f"Erreur analyse sémantique: {e}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        })


@app.route('/run-analysis/sentiment')
def run_sentiment_analysis():
    """Lance l'analyse de sentiment"""
    try:
        # Trouver le dernier corpus prétraité
        preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
        if not preprocessed_files:
            return jsonify({
                "success": False,
                "message": "Aucun corpus prétraité trouvé"
            })
        
        corpus_path = max(preprocessed_files)
        
        # Importer et exécuter l'analyseur de sentiment
        from analysis.sentiment_analyzer import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        analyzer.load_corpus(corpus_path)
        results = analyzer.analyze_all()
        
        # Recharger les résultats dans l'interface
        global ANALYSIS_RESULTS
        ANALYSIS_RESULTS["sentiment"] = results
        
        # Recharger tous les résultats pour synchronisation
        load_analysis_results()
        
        return jsonify({
            "success": True,
            "message": f"Analyse de sentiment terminée.",
            "redirect": "/analysis#sentiment"
        })
        
    except Exception as e:
        logger.error(f"Erreur analyse sentiment: {e}")
        return jsonify({
            "success": False,
            "message": f"Erreur: {str(e)}"
        })


@app.route('/api/visualizations')
def api_visualizations():
    """API pour récupérer la liste des visualisations"""
    
    viz_dir = "visualizations"
    visualizations = []
    
    if os.path.exists(viz_dir):
        for file in os.listdir(viz_dir):
            if file.endswith(('.png', '.jpg', '.jpeg', '.gif', '.html')):
                file_type = 'interactive' if file.endswith('.html') else 'image'
                visualizations.append({
                    'name': file.replace('.html', '').replace('.png', '').replace('.jpg', '').replace('.jpeg', '').replace('.gif', '').replace('_', ' ').title(),
                    'file': file,
                    'type': file_type,
                    'url': f'/viz/{file}'
                })
    
    return jsonify({
        'visualizations': visualizations,
        'total': len(visualizations)
    })


# Initialisation au démarrage
@app.before_request
def initialize():
    """Initialise l'application si nécessaire"""
    
    if not CONFIG:
        load_config()
    
    if not CORPUS_DATA:
        load_corpus_data()
    
    if not ANALYSIS_RESULTS:
        load_analysis_results()
        
# @app.route('/run-analysis/semantic')
# def run_semantic_analysis():
#     """Lance l'analyse sémantique"""
#     try:
#         # Trouver le dernier corpus prétraité
#         preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
#         if not preprocessed_files:
#             return jsonify({
#                 "success": False,
#                 "message": "Aucun corpus prétraité trouvé"
#             })
        
#         corpus_path = max(preprocessed_files)
        
#         # Importer et exécuter l'analyseur sémantique
#         from analysis.semantic_analyzer import EnhancedSemanticAnalyzer
        
#         analyzer = EnhancedSemanticAnalyzer()
#         analyzer.load_corpus(corpus_path)
#         results = analyzer.analyze_all()
        
#         # Recharger les résultats dans l'interface
#         global ANALYSIS_RESULTS
#         ANALYSIS_RESULTS["semantic"] = results
        
#         return jsonify({
#             "success": True,
#             "message": f"Analyse sémantique terminée. {len(results.get('semantic_fields', {}))} topics analysés.",
#             "redirect": "/analysis#semantic"
#         })
        
#     except Exception as e:
#         logger.error(f"Erreur analyse sémantique: {e}")
#         return jsonify({
#             "success": False,
#             "message": f"Erreur: {str(e)}"
#         })

if __name__ == '__main__':
    # Charger la configuration avant le premier démarrage
    load_config()
    load_corpus_data()
    load_analysis_results()
    
    logger.info("=" * 60)
    logger.info("NLP Bias Analyzer - Interface Web")
    logger.info("=" * 60)
    logger.info(f"Articles chargés: {len(CORPUS_DATA)}")
    
    # Vérifier quelles analyses sont disponibles
    available_analyses = []
    if ANALYSIS_RESULTS.get('lexical'):
        available_analyses.append("Lexicale")
    if ANALYSIS_RESULTS.get('semantic'):
        available_analyses.append("Sémantique")
    if ANALYSIS_RESULTS.get('sentiment'):
        available_analyses.append("Sentiment")
    
    if available_analyses:
        logger.info(f"Analyses disponibles: {', '.join(available_analyses)}")
    else:
        logger.info("Aucune analyse disponible. Lancez une analyse depuis l'interface web.")
    
    # Démarrer l'application
    host = CONFIG.get('web_interface', {}).get('host', '127.0.0.1')
    port = CONFIG.get('web_interface', {}).get('port', 5000)
    
    logger.info(f"Interface web accessible sur: http://{host}:{port}")
    logger.info("=" * 60)
    
    app.run(
        host=host,
        port=port,
        debug=False,
        use_reloader=False
    )