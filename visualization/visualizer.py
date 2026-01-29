"""
Module de visualisation pour les résultats d'analyse
Génère des graphiques et des visualisations interactives
"""

import json
import os
from typing import Dict, List, Tuple
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import yaml
from tqdm import tqdm
import logging

# Configuration matplotlib pour les graphiques statiques
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Visualizer:
    """Générateur de visualisations pour l'analyse NLP"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialise le visualiseur
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.viz_config = self.config['visualization']
        self.color_palette = self.viz_config.get('color_palette', 'Set2')
        
        # Couleurs pour les topics
        self.topic_colors = {
            'gaza': '#d62728',      # Rouge
            'ukraine': '#2ca02c',   # Vert
            'default': '#1f77b4'    # Bleu
        }
        
        # Résultats
        self.results = {}
    
    def load_results(self, results_dir="analysis_results"):
        """
        Charge tous les fichiers de résultats
        
        Args:
            results_dir: Répertoire contenant les résultats
        """
        
        import glob
        
        # Trouver tous les fichiers de résultats
        result_files = glob.glob(f"{results_dir}/*.json")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Identifier le type d'analyse
                if 'lexical' in file_path:
                    self.results['lexical'] = data
                elif 'semantic' in file_path:
                    self.results['semantic'] = data
                elif 'sentiment' in file_path:
                    self.results['sentiment'] = data
                
                logger.info(f"Résultats chargés: {file_path}")
            
            except Exception as e:
                logger.warning(f"Erreur lors du chargement de {file_path}: {e}")
        
        logger.info(f"Types de résultats chargés: {list(self.results.keys())}")
    
    def generate_all_visualizations(self, output_dir="visualizations"):
        """
        Génère toutes les visualisations
        
        Args:
            output_dir: Répertoire de sortie
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info("Génération de toutes les visualisations...")
        
        # 1. Visualisations lexicales
        if 'lexical' in self.results:
            self._generate_lexical_visualizations(output_dir)
        
        # 2. Visualisations sémantiques
        if 'semantic' in self.results:
            self._generate_semantic_visualizations(output_dir)
        
        # 3. Visualisations de sentiment
        if 'sentiment' in self.results:
            self._generate_sentiment_visualizations(output_dir)
        
        # 4. Tableau de bord comparatif
        self._generate_comparative_dashboard(output_dir)
        
        logger.info(f"Toutes les visualisations sauvegardées dans: {output_dir}")
    
    def _generate_lexical_visualizations(self, output_dir: str):
        """Génère les visualisations lexicales"""
        
        lexical_results = self.results['lexical']
        
        # 1. Fréquences de mots
        self._plot_word_frequencies(lexical_results['word_frequencies'], output_dir)
        
        # 2. Comparaison TF-IDF
        self._plot_tfidf_comparison(lexical_results['tfidf_analysis'], output_dir)
        
        # 3. Patterns de biais
        self._plot_bias_patterns(lexical_results['bias_patterns'], output_dir)
        
        # 4. Réseau de cooccurrences
        self._plot_cooccurrence_network(lexical_results['cooccurrences'], output_dir)
        
        # 5. Comparaison lexicale
        self._plot_lexical_comparison(lexical_results['lexical_comparison'], output_dir)
    
    def _generate_semantic_visualizations(self, output_dir: str):
        """Génère les visualisations sémantiques"""
        
        semantic_results = self.results['semantic']
        
        # 1. Champs sémantiques
        self._plot_semantic_fields(semantic_results['semantic_fields'], output_dir)
        
        # 2. Concordance
        self._plot_concordance_heatmap(semantic_results['concordance'], output_dir)
        
        # 3. Associations de mots
        self._plot_word_associations(semantic_results['associations'], output_dir)
    
    def _generate_sentiment_visualizations(self, output_dir: str):
        """Génère les visualisations de sentiment"""
        
        sentiment_results = self.results['sentiment']
        
        # 1. Distribution du sentiment
        self._plot_sentiment_distribution(sentiment_results['topic_sentiment'], output_dir)
        
        # 2. Comparaison de sentiment
        self._plot_sentiment_comparison(sentiment_results['sentiment_comparison'], output_dir)
        
        # 3. Sentiment par mot-cible
        self._plot_target_word_sentiment(sentiment_results['target_word_sentiment'], output_dir)
        
        # 4. Évolution temporelle
        if 'temporal_sentiment' in sentiment_results:
            self._plot_temporal_sentiment(sentiment_results['temporal_sentiment'], output_dir)
    
    def _plot_word_frequencies(self, word_freq: Dict, output_dir: str):
        """Visualise les fréquences de mots"""
        
        for topic, frequencies in word_freq.items():
            if not frequencies:
                continue
            
            # Préparer les données
            words = [item[0] for item in frequencies[:20]]
            counts = [item[1] for item in frequencies[:20]]
            
            # Créer le graphique
            fig = go.Figure(data=[
                go.Bar(
                    x=counts,
                    y=words,
                    orientation='h',
                    marker_color=self.topic_colors.get(topic, self.topic_colors['default']),
                    text=counts,
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=f'Top 20 mots les plus fréquents - {topic.upper()}',
                xaxis_title='Fréquence',
                yaxis_title='Mots',
                height=600,
                yaxis={'categoryorder': 'total ascending'},
                font=dict(size=12)
            )
            
            # Sauvegarder
            output_path = os.path.join(output_dir, f'word_frequencies_{topic}.html')
            fig.write_html(output_path)
            
            # Version matplotlib pour le rapport PDF
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(words)), counts, color=self.topic_colors.get(topic, 'blue'))
            plt.yticks(range(len(words)), words)
            plt.xlabel('Fréquence')
            plt.title(f'Top 20 mots les plus fréquents - {topic.upper()}')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_dir, f'word_frequencies_{topic}.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _plot_tfidf_comparison(self, tfidf_results: Dict, output_dir: str):
        """Visualise la comparaison TF-IDF"""
        
        # Créer un graphique comparatif
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=list(tfidf_results.keys())[:2] if len(tfidf_results) >= 2 else list(tfidf_results.keys())
        )
        
        for i, (topic, scores) in enumerate(list(tfidf_results.items())[:2]):
            if not scores:
                continue
            
            terms = [item[0] for item in scores[:15]]
            tfidf_scores = [item[1] for item in scores[:15]]
            
            fig.add_trace(
                go.Bar(
                    x=tfidf_scores,
                    y=terms,
                    orientation='h',
                    name=topic,
                    marker_color=self.topic_colors.get(topic, self.topic_colors['default'])
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='Comparaison TF-IDF - Termes les plus caractéristiques',
            height=600,
            showlegend=False
        )
        
        output_path = os.path.join(output_dir, 'tfidf_comparison.html')
        fig.write_html(output_path)
    
    def _plot_bias_patterns(self, bias_patterns: Dict, output_dir: str):
        """Visualise les patterns de biais"""
        
        # 1. Termes déshumanisants vs humanisants
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Déshumanisants
        dehumanizing_data = bias_patterns['dehumanizing_terms']
        self._plot_bias_category(axes[0, 0], dehumanizing_data, 'Termes Déshumanisants', 'red')
        
        # Humanisants
        humanizing_data = bias_patterns['humanizing_terms']
        self._plot_bias_category(axes[0, 1], humanizing_data, 'Termes Humanisants', 'green')
        
        # Ton émotionnel
        emotional_data = bias_patterns['emotional_tone']
        self._plot_emotional_tone(axes[1, 0], emotional_data)
        
        # Euphémismes vs termes directs
        euphemism_data = bias_patterns['euphemisms_vs_direct']
        self._plot_euphemism_ratio(axes[1, 1], euphemism_data)
        
        plt.suptitle('Analyse des Patterns de Biais Lexicaux', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, 'bias_patterns.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_bias_category(self, ax, data: Dict, title: str, color: str):
        """Visualise une catégorie de biais"""
        
        topics = list(data.keys())
        categories = list(data[topics[0]].keys()) if topics else []
        
        x = np.arange(len(categories))
        width = 0.35
        
        for i, topic in enumerate(topics):
            values = [data[topic][cat] for cat in categories]
            ax.bar(x + i * width, values, width, label=topic, 
                  color=self.topic_colors.get(topic, color))
        
        ax.set_xlabel('Catégories')
        ax.set_ylabel('Fréquence')
        ax.set_title(title)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend()
    
    def _plot_emotional_tone(self, ax, data: Dict):
        """Visualise le ton émotionnel"""
        
        topics = list(data.keys())
        tones = ['positive', 'negative', 'neutral']
        
        x = np.arange(len(topics))
        width = 0.25
        
        colors = ['green', 'red', 'gray']
        
        for i, tone in enumerate(tones):
            values = [data[topic].get(tone, 0) for topic in topics]
            ax.bar(x + i * width, values, width, label=tone, color=colors[i])
        
        ax.set_xlabel('Topics')
        ax.set_ylabel('Fréquence')
        ax.set_title('Ton Émotionnel')
        ax.set_xticks(x + width)
        ax.set_xticklabels(topics)
        ax.legend()
    
    def _plot_euphemism_ratio(self, ax, data: Dict):
        """Visualise le ratio euphémismes/termes directs"""
        
        topics = list(data.keys())
        
        euphemisms = [data[topic]['euphemisms'] for topic in topics]
        direct_terms = [data[topic]['direct_terms'] for topic in topics]
        
        x = np.arange(len(topics))
        width = 0.35
        
        ax.bar(x - width/2, euphemisms, width, label='Euphémismes', color='orange')
        ax.bar(x + width/2, direct_terms, width, label='Termes Directs', color='purple')
        
        ax.set_xlabel('Topics')
        ax.set_ylabel('Fréquence')
        ax.set_title('Euphémismes vs Termes Directs')
        ax.set_xticks(x)
        ax.set_xticklabels(topics)
        ax.legend()
    
    def _plot_cooccurrence_network(self, cooccurrences: Dict, output_dir: str):
        """Visualise le réseau de cooccurrences"""
        
        for topic, cooccur_data in cooccurrences.items():
            if not cooccur_data:
                continue
            
            # Créer le graphe
            G = nx.Graph()
            
            # Ajouter les nœuds et les arêtes
            for word, related_words in list(cooccur_data.items())[:20]:  # Limiter pour la clarté
                G.add_node(word)
                
                for related_word, count in related_words[:5]:  # Top 5 connexions
                    G.add_node(related_word)
                    G.add_edge(word, related_word, weight=count)
            
            # Calculer la disposition
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Créer les traces pour Plotly
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Taille basée sur le degré
                node_size.append(len(list(G.neighbors(node))) * 10 + 10)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                hovertext=[f"{node}<br>Connexions: {len(list(G.neighbors(node)))}" for node in G.nodes()],
                marker=dict(
                    size=node_size,
                    color=self.topic_colors.get(topic, self.topic_colors['default']),
                    line=dict(width=2, color='white')
                )
            )
            
            # Créer la figure
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Reseau de Cooccurrences - {topic.upper()}',
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        text="Les mots sont connectés s'ils apparaissent fréquemment ensemble",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color="gray", size=12)
                    )],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            # Sauvegarder
            output_path = os.path.join(output_dir, f'cooccurrence_network_{topic}.html')
            fig.write_html(output_path)
    
    def _plot_lexical_comparison(self, comparison_data: Dict, output_dir: str):
        """Visualise la comparaison lexicale"""
        
        # Similarité cosinus
        similarities = comparison_data['cosine_similarity']
        
        if similarities:
            pairs = list(similarities.keys())
            values = list(similarities.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=pairs,
                    y=values,
                    marker_color='steelblue',
                    text=[f'{v:.3f}' for v in values],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title='Similarité Cosinus entre les Vocabulaires',
                xaxis_title='Paires de Topics',
                yaxis_title='Similarité Cosinus (0-1)',
                yaxis=dict(range=[0, 1])
            )
            
            output_path = os.path.join(output_dir, 'cosine_similarity.html')
            fig.write_html(output_path)
    
    def _plot_semantic_fields(self, semantic_fields: Dict, output_dir: str):
        """Visualise les champs sémantiques"""
        
        for topic, fields in semantic_fields.items():
            # Nuage de mots basé sur TF-IDF
            fig = go.Figure()
            
            for category, words in fields.items():
                if not words:
                    continue
                
                terms = [item[0] for item in words[:30]]
                scores = [item[1] for item in words[:30]]
                
                fig.add_trace(go.Scatter(
                    x=list(range(len(terms))),
                    y=scores,
                    mode='markers+text',
                    text=terms,
                    textposition="top center",
                    name=category,
                    marker=dict(
                        size=[s*100 + 10 for s in scores],
                        opacity=0.7
                    )
                ))
            
            fig.update_layout(
                title=f'Champs Sémantiques - {topic.upper()}',
                xaxis_title='Rang',
                yaxis_title='Score TF-IDF',
                height=600
            )
            
            output_path = os.path.join(output_dir, f'semantic_fields_{topic}.html')
            fig.write_html(output_path)
    
    def _plot_concordance_heatmap(self, concordance_data: Dict, output_dir: str):
        """Visualise la concordance sous forme de heatmap"""
        
        # Créer une matrice de contexte
        for topic, words in concordance_data.items():
            if not words:
                continue
            
            # Collecter tous les mots de contexte
            all_context_words = set()
            target_words = []
            
            for target_word, contexts in list(words.items())[:10]:  # Limiter pour la clarté
                if not contexts:
                    continue
                
                target_words.append(target_word)
                
                for context in contexts:
                    before_words = context.get('context_before', '').split()
                    after_words = context.get('context_after', '').split()
                    all_context_words.update(before_words + after_words)
            
            if not target_words or not all_context_words:
                continue
            
            # Filtrer les mots de contexte
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
            context_words = [word for word in all_context_words if len(word) > 2 and word.lower() not in stop_words][:50]
            
            # Créer la matrice
            matrix = np.zeros((len(target_words), len(context_words)))
            
            for i, target_word in enumerate(target_words):
                contexts = words.get(target_word, [])
                
                for context in contexts:
                    before = context.get('context_before', '').lower().split()
                    after = context.get('context_after', '').lower().split()
                    context_tokens = before + after
                    
                    for j, context_word in enumerate(context_words):
                        matrix[i, j] += context_tokens.count(context_word.lower())
            
            # Normaliser
            matrix = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)
            
            # Créer la heatmap
            fig = go.Figure(data=go.Heatmap(
                z=matrix,
                x=context_words,
                y=target_words,
                colorscale='Blues',
                hoverongaps=False
            ))
            
            fig.update_layout(
                title=f'Carte de Chaleur des Contextes - {topic.upper()}',
                xaxis_title='Mots de Contexte',
                yaxis_title='Mots Cibles',
                height=800
            )
            
            output_path = os.path.join(output_dir, f'concordance_heatmap_{topic}.html')
            fig.write_html(output_path)
    
    def _plot_word_associations(self, associations_data: Dict, output_dir: str):
        """Visualise les associations de mots"""
        
        for topic, word_associations in associations_data.items():
            # Créer un graphique de réseau
            G = nx.Graph()
            
            for target_word, associations in list(word_associations.items())[:5]:  # Limiter pour la clarté
                if not associations:
                    continue
                
                G.add_node(target_word, size=20)
                
                for assoc_word, score in associations[:5]:
                    G.add_node(assoc_word, size=score*100)
                    G.add_edge(target_word, assoc_word, weight=score)
            
            if G.number_of_edges() == 0:
                continue
            
            # Layout
            pos = nx.spring_layout(G, k=2, iterations=20)
            
            # Traces Plotly
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(G[edge[0]][edge[1]]['weight'])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='gray'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
                # Taille basée sur le nombre de connexions ou le score
                if 'size' in G.nodes[node]:
                    node_size.append(G.nodes[node]['size'])
                else:
                    node_size.append(len(list(G.neighbors(node))) * 10 + 10)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=node_size,
                    color=self.topic_colors.get(topic, self.topic_colors['default']),
                    line=dict(width=2, color='white')
                )
            )
            
            fig = go.Figure(
                data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=f'Associations Sémantiques - {topic.upper()}',
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                )
            )
            
            output_path = os.path.join(output_dir, f'word_associations_{topic}.html')
            fig.write_html(output_path)
    
    def _plot_sentiment_distribution(self, topic_sentiment: Dict, output_dir: str):
        """Visualise la distribution du sentiment"""
        
        # Graphique en barres empilées
        topics = list(topic_sentiment.keys())
        
        fig = go.Figure()
        
        sentiments = ['positive', 'negative', 'neutral']
        colors = ['green', 'red', 'gray']
        
        for i, sentiment in enumerate(sentiments):
            values = []
            for topic in topics:
                distribution = topic_sentiment[topic]['consensus_distribution']
                total = sum(distribution.values())
                value = distribution.get(sentiment, 0) / total * 100 if total > 0 else 0
                values.append(value)
            
            fig.add_trace(go.Bar(
                name=sentiment,
                x=topics,
                y=values,
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            title='Distribution du Sentiment par Topic',
            xaxis_title='Topics',
            yaxis_title='Pourcentage (%)',
            barmode='stack',
            height=500
        )
        
        output_path = os.path.join(output_dir, 'sentiment_distribution.html')
        fig.write_html(output_path)
        
        # Version matplotlib
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(topics))
        width = 0.25
        
        for i, sentiment in enumerate(sentiments):
            values = []
            for topic in topics:
                distribution = topic_sentiment[topic]['consensus_distribution']
                total = sum(distribution.values())
                value = distribution.get(sentiment, 0) / total * 100 if total > 0 else 0
                values.append(value)
            
            plt.bar(x + i * width, values, width, label=sentiment, color=colors[i])
        
        plt.xlabel('Topics')
        plt.ylabel('Pourcentage (%)')
        plt.title('Distribution du Sentiment par Topic')
        plt.xticks(x + width, topics)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'sentiment_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sentiment_comparison(self, comparison_data: Dict, output_dir: str):
        """Visualise la comparaison de sentiment"""
        
        for comparison, data in comparison_data.items():
            labels = list(data.keys())
            
            fig = go.Figure()
            
            # Extraire les pourcentages pour chaque topic
            topic1_percentages = [data[label][f'{comparison.split("_vs_")[0]}_percent'] for label in labels]
            topic2_percentages = [data[label][f'{comparison.split("_vs_")[1]}_percent'] for label in labels]
            
            fig.add_trace(go.Bar(
                name=comparison.split('_vs_')[0],
                x=labels,
                y=topic1_percentages,
                marker_color='blue'
            ))
            
            fig.add_trace(go.Bar(
                name=comparison.split('_vs_')[1],
                x=labels,
                y=topic2_percentages,
                marker_color='red'
            ))
            
            fig.update_layout(
                title=f'Comparaison de Sentiment - {comparison}',
                xaxis_title='Labels de Sentiment',
                yaxis_title='Pourcentage (%)',
                barmode='group',
                height=500
            )
            
            output_path = os.path.join(output_dir, f'sentiment_comparison_{comparison}.html')
            fig.write_html(output_path)
    
    def _plot_target_word_sentiment(self, target_sentiment: Dict, output_dir: str):
        """Visualise le sentiment par mot-cible"""
        
        for topic, words in target_sentiment.items():
            if not words:
                continue
            
            fig = go.Figure()
            
            for word, data in words.items():
                distribution = data['aggregated']['distribution']
                labels = list(distribution.keys())
                values = list(distribution.values())
                
                fig.add_trace(go.Bar(
                    name=word,
                    x=labels,
                    y=values
                ))
            
            fig.update_layout(
                title=f'Sentiment par Mot-Cible - {topic.upper()}',
                xaxis_title='Sentiment',
                yaxis_title='Nombre de Contextes',
                barmode='group',
                height=600
            )
            
            output_path = os.path.join(output_dir, f'target_word_sentiment_{topic}.html')
            fig.write_html(output_path)
    
    def _plot_temporal_sentiment(self, temporal_data: Dict, output_dir: str):
        """Visualise l'évolution temporelle du sentiment"""
        
        for topic, monthly_data in temporal_data.items():
            if not monthly_data:
                continue
            
            # Trier par date
            sorted_months = sorted(monthly_data.keys())
            
            fig = go.Figure()
            
            # Pour chaque sentiment
            sentiments = ['positive', 'negative', 'neutral']
            colors = ['green', 'red', 'gray']
            
            for i, sentiment in enumerate(sentiments):
                percentages = []
                
                for month in sorted_months:
                    distribution = monthly_data[month]['sentiment_distribution']
                    total = sum(distribution.values())
                    pct = distribution.get(sentiment, 0) / total * 100 if total > 0 else 0
                    percentages.append(pct)
                
                fig.add_trace(go.Scatter(
                    x=sorted_months,
                    y=percentages,
                    mode='lines+markers',
                    name=sentiment,
                    line=dict(color=colors[i], width=3),
                    marker=dict(size=8)
                ))
            
            fig.update_layout(
                title=f'Évolution Temporelle du Sentiment - {topic.upper()}',
                xaxis_title='Date',
                yaxis_title='Pourcentage (%)',
                height=500,
                hovermode='x unified'
            )
            
            output_path = os.path.join(output_dir, f'temporal_sentiment_{topic}.html')
            fig.write_html(output_path)
    
    def _generate_comparative_dashboard(self, output_dir: str):
        """Génère un tableau de bord comparatif"""
        
        # Créer un dashboard avec plusieurs graphiques
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Distribution du Sentiment',
                'Top 10 Mots (Gaza)',
                'Top 10 Mots (Ukraine)',
                'Ratio Euphémismes/Termes Directs'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Distribution du sentiment
        if 'sentiment' in self.results:
            topic_sentiment = self.results['sentiment']['topic_sentiment']
            topics = list(topic_sentiment.keys())
            
            for i, sentiment in enumerate(['positive', 'negative', 'neutral']):
                values = []
                for topic in topics:
                    distribution = topic_sentiment[topic]['consensus_distribution']
                    total = sum(distribution.values())
                    value = distribution.get(sentiment, 0) / total * 100 if total > 0 else 0
                    values.append(value)
                
                colors = ['green', 'red', 'gray']
                fig.add_trace(
                    go.Bar(x=topics, y=values, name=sentiment, marker_color=colors[i]),
                    row=1, col=1
                )
        
        # 2 et 3. Top mots
        if 'lexical' in self.results:
            word_freq = self.results['lexical']['word_frequencies']
            
            for i, (topic, frequencies) in enumerate(list(word_freq.items())[:2]):
                if not frequencies:
                    continue
                
                words = [item[0] for item in frequencies[:10]]
                counts = [item[1] for item in frequencies[:10]]
                
                fig.add_trace(
                    go.Bar(x=counts, y=words, orientation='h', name=f'Top {topic}', 
                          marker_color=self.topic_colors.get(topic, self.topic_colors['default']),
                          showlegend=False),
                    row=1 if i == 0 else 2, col=1 if i == 0 else 1
                )
        
        # 4. Ratio euphémismes
        if 'lexical' in self.results:
            euphemism_data = self.results['lexical']['bias_patterns']['euphemisms_vs_direct']
            
            topics = list(euphemism_data.keys())
            euphemisms = [euphemism_data[topic]['euphemisms'] for topic in topics]
            direct_terms = [euphemism_data[topic]['direct_terms'] for topic in topics]
            
            fig.add_trace(
                go.Bar(x=topics, y=euphemisms, name='Euphémismes', marker_color='orange'),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Bar(x=topics, y=direct_terms, name='Termes Directs', marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Tableau de Bord Comparatif - Analyse des Biais Médiatiques",
            height=800,
            showlegend=True
        )
        
        output_path = os.path.join(output_dir, 'comparative_dashboard.html')
        fig.write_html(output_path)


def main():
    """Fonction principale de test"""
    
    # Initialiser le visualiseur
    visualizer = Visualizer()
    
    # Charger les résultats
    visualizer.load_results("analysis_results")
    
    # Générer les visualisations
    visualizer.generate_all_visualizations("visualizations")
    
    print("Visualisations générées avec succès!")


if __name__ == "__main__":
    main()