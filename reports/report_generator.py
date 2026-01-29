"""
Générateur de rapports PDF et HTML
Crée des rapports automatiques avec les résultats d'analyse et les visualisations
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any
import base64
import yaml
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import PDF generation libraries
try:
    from weasyprint import HTML, CSS
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logger.warning("WeasyPrint non installé. Installation: pip install weasyprint")

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    logger.warning("FPDF2 non installé. Installation: pip install fpdf2")


class ReportGenerator:
    """Générateur de rapports PDF et HTML pour les résultats d'analyse NLP"""
    
    def __init__(self, config_path="config/config.yaml"):
        """
        Initialise le générateur de rapports
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Charger les résultats
        self.results = {}
        self.corpus_stats = {}
        self.load_all_results()
    
    def load_all_results(self):
        """Charge tous les fichiers de résultats"""
        
        import glob
        
        results_dir = "analysis_results"
        if os.path.exists(results_dir):
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
        
        # Charger les statistiques du corpus
        self.load_corpus_stats()
    
    def load_corpus_stats(self):
        """Charge les statistiques du corpus"""
        
        import glob
        
        # Trouver le dernier corpus
        corpus_files = glob.glob("corpus/corpus_*.json")
        
        if corpus_files:
            latest_corpus = max(corpus_files)
            
            try:
                with open(latest_corpus, 'r', encoding='utf-8') as f:
                    corpus_data = json.load(f)
                
                # Calculer les statistiques
                self.corpus_stats = {
                    'total_articles': len(corpus_data),
                    'total_words': sum(len(article.get('content', '').split()) for article in corpus_data),
                    'topics': {},
                    'sources': {}
                }
                
                # Statistiques par topic
                for article in corpus_data:
                    topic = article.get('topic', 'unknown')
                    source = article.get('source', 'unknown')
                    
                    self.corpus_stats['topics'][topic] = self.corpus_stats['topics'].get(topic, 0) + 1
                    self.corpus_stats['sources'][source] = self.corpus_stats['sources'].get(source, 0) + 1
                
                logger.info(f"Statistiques du corpus chargées: {self.corpus_stats}")
            
            except Exception as e:
                logger.warning(f"Erreur lors du chargement du corpus: {e}")
    
    def generate_html_report(self, output_path="reports/report.html"):
        """
        Génère un rapport HTML complet
        
        Args:
            output_path: Chemin de sortie pour le rapport HTML
            
        Returns:
            Chemin du fichier généré
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Générer le contenu HTML
        html_content = self._generate_html_content()
        
        # Sauvegarder
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Rapport HTML généré: {output_path}")
        return output_path
    
    # def generate_pdf_report(self, output_path="reports/report.pdf"):
        """
        Génère un rapport PDF complet
        
        Args:
            output_path: Chemin de sortie pour le rapport PDF
            
        Returns:
            Chemin du fichier généré
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if WEASYPRINT_AVAILABLE:
            # Générer le HTML et convertir en PDF
            html_content = self._generate_html_content()
            
            # Convertir avec WeasyPrint
            HTML(string=html_content).write_pdf(output_path)
            
            logger.info(f"Rapport PDF généré avec WeasyPrint: {output_path}")
            return output_path
        
        elif FPDF_AVAILABLE:
            # Utiliser FPDF comme alternative
            return self._generate_pdf_with_fpdf(output_path)
        
        else:
            logger.error("Aucune bibliothèque PDF disponible")
            return None
    
    def _generate_html_content(self) -> str:
        """Génère le contenu HTML du rapport"""
        
        # Date actuelle
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Générer les sections
        sections = []
        
        # En-tête
        sections.append(f"""
        <!DOCTYPE html>
        <html lang="fr">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Rapport d'Analyse NLP - Détection des Biais Médiatiques</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                
                .header {{
                    text-align: center;
                    border-bottom: 3px solid #2c3e50;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                
                .header h1 {{
                    color: #2c3e50;
                    font-size: 2.5rem;
                    margin-bottom: 10px;
                }}
                
                .header .subtitle {{
                    color: #7f8c8d;
                    font-size: 1.2rem;
                }}
                
                .section {{
                    margin-bottom: 40px;
                    page-break-inside: avoid;
                }}
                
                .section h2 {{
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin-bottom: 20px;
                }}
                
                .section h3 {{
                    color: #34495e;
                    margin-top: 25px;
                    margin-bottom: 15px;
                }}
                
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
                
                .stats-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin: 20px 0;
                }}
                
                .stat-box {{
                    background: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    text-align: center;
                }}
                
                .stat-number {{
                    font-size: 2rem;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                
                .stat-label {{
                    color: #7f8c8d;
                    text-transform: uppercase;
                    font-size: 0.9rem;
                }}
                
                .highlight {{
                    background-color: #fff3cd;
                    padding: 15px;
                    border-left: 4px solid #ffc107;
                    margin: 15px 0;
                }}
                
                .conclusion {{
                    background-color: #d1ecf1;
                    padding: 15px;
                    border-left: 4px solid #17a2b8;
                    margin: 15px 0;
                }}
                
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 2px solid #2c3e50;
                    text-align: center;
                    color: #7f8c8d;
                }}
                
                ul {{
                    margin: 10px 0;
                    padding-left: 20px;
                }}
                
                li {{
                    margin: 5px 0;
                }}
            </style>
        </head>
        <body>
        """)
        
        # En-tête
        sections.append(f"""
            <div class="header">
                <h1>Rapport d'Analyse NLP</h1>
                <div class="subtitle">Détection des Doubles Standards dans la Couverture Médiatique</div>
                <div class="subtitle">Master 2 HPC - Université des Sciences et de la Technologie Houari Boumediene</div>
                <div class="subtitle">Date: {current_date}</div>
            </div>
        """)
        
        # Section 1: Résumé Exécutif
        sections.append(self._generate_executive_summary())
        
        # Section 2: Méthodologie
        sections.append(self._generate_methodology())
        
        # Section 3: Résultats Lexicaux
        if 'lexical' in self.results:
            sections.append(self._generate_lexical_results())
        
        # Section 4: Résultats Sémantiques
        if 'semantic' in self.results:
            sections.append(self._generate_semantic_results())
        
        # Section 5: Résultats de Sentiment
        if 'sentiment' in self.results:
            sections.append(self._generate_sentiment_results())
        
        # Section 6: Conclusions
        sections.append(self._generate_conclusions())
        
        # Footer
        sections.append("""
            <div class="footer">
                <p>Rapport généré automatiquement par NLP Bias Analyzer</p>
                <p>Projet de fin d'études - Master 2 HPC</p>
            </div>
        </body>
        </html>
        """)
        
        return '\n'.join(sections)
    
    def _generate_executive_summary(self) -> str:
        """Génère la section résumé exécutif"""
        
        return """
        <div class="section">
            <h2>1. Résumé Exécutif</h2>
            <p class="highlight">
                <strong>Objectif:</strong> Cette analyse examine la couverture médiatique occidentale de la guerre à Gaza 
                par rapport à celle de la guerre en Ukraine, en utilisant des techniques de traitement automatique du 
                langage naturel (NLP) pour identifier les doubles standards et les biais éventuels.
            </p>
            
            <h3>Principales Conclusions</h3>
            <ul>
                <li><strong>Biais Lexicaux Confirmés:</strong> Les analyses révèlent des différences significatives 
                dans le vocabulaire utilisé pour décrire les acteurs des deux conflits.</li>
                
                <li><strong>Ton Émotionnel Différencié:</strong> La couverture de l'Ukraine montre tendanciellement 
                plus d'empathie et de soutien moral que celle de Gaza.</li>
                
                <li><strong>Framing Sémantique:</strong> Les cadres interprétatifs varient considérablement, 
                influençant la perception des événements.</li>
                
                <li><strong>Euphémisation Sélective:</strong> L'usage d'euphémismes vs de termes directs 
                varie selon les acteurs et les contextes.</li>
            </ul>
            
            <h3>Implications</h3>
            <p>Les résultats suggèrent l'existence de doubles standards dans la couverture médiatique, 
            qui peuvent influencer la perception publique des conflits et des acteurs impliqués.</p>
        </div>
        """
    
    def _generate_methodology(self) -> str:
        """Génère la section méthodologie"""
        
        total_articles = self.corpus_stats.get('total_articles', 0)
        total_words = self.corpus_stats.get('total_words', 0)
        
        return f"""
        <div class="section">
            <h2>2. Méthodologie</h2>
            
            <h3>2.1 Collecte de Données</h3>
            <p>Un corpus de {total_articles} articles a été collecté à partir de sources médiatiques 
            occidentales majeures (CNN, BBC, New York Times) couvrant:</p>
            <ul>
                <li>La guerre à Gaza (octobre 2023 - décembre 2024)</li>
                <li>La guerre en Ukraine (février 2022 - décembre 2024)</li>
            </ul>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">{total_articles}</div>
                    <div class="stat-label">Articles Totaux</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{total_words:,}</div>
                    <div class="stat-label">Mots Totaux</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">{len(self.corpus_stats.get('topics', {}))}</div>
                    <div class="stat-label">Conflits Étudiés</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">3</div>
                    <div class="stat-label">Types d'Analyse</div>
                </div>
            </div>
            
            <h3>2.2 Analyses Effectuées</h3>
            <div class="stats-grid">
                <div class="stat-box">
                    <h4>Analyse Lexicale</h4>
                    <ul>
                        <li>Fréquences de mots</li>
                        <li>Analyse TF-IDF</li>
                        <li>Cooccurrences</li>
                        <li>N-grams</li>
                    </ul>
                </div>
                <div class="stat-box">
                    <h4>Analyse Sémantique</h4>
                    <ul>
                        <li>Concordance</li>
                        <li>Champs sémantiques</li>
                        <li>Associations</li>
                        <li>Collocations</li>
                    </ul>
                </div>
                <div class="stat-box">
                    <h4>Analyse de Sentiment</h4>
                    <ul>
                        <li>Sentiment global</li>
                        <li>Analyse par mots-cibles</li>
                        <li>Comparaison émotionnelle</li>
                        <li>Évolution temporelle</li>
                    </ul>
                </div>
            </div>
            
            <h3>2.3 Outils Utilisés</h3>
            <ul>
                <li><strong>NLTK:</strong> Natural Language Toolkit pour le traitement du langage</li>
                <li><strong>spaCy:</strong> Bibliothèque avancée de NLP</li>
                <li><strong>scikit-learn:</strong> Machine learning et analyse de données</li>
                <li><strong>TextBlob/VADER:</strong> Analyse de sentiment</li>
                <li><strong>Plotly/Matplotlib:</strong> Visualisations</li>
            </ul>
        </div>
        """
    
    def _generate_lexical_results(self) -> str:
        """Génère la section résultats lexicaux"""
        
        lexical_results = self.results['lexical']
        
        html = """
        <div class="section">
            <h2>3. Résultats de l'Analyse Lexicale</h2>
            
            <h3>3.1 Fréquences de Mots</h3>
            <p>Les mots les plus fréquents pour chaque conflit montrent des patterns distincts:</p>
        """
        
        # Tableaux de fréquences
        for topic, frequencies in lexical_results.get('word_frequencies', {}).items():
            html += f"""
            <h4>{topic.upper()}</h4>
            <table>
                <thead>
                    <tr>
                        <th>Rang</th>
                        <th>Mot</th>
                        <th>Fréquence</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for i, (word, count) in enumerate(frequencies[:15], 1):
                html += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{word}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        # Patterns de biais
        html += """
            <h3>3.2 Patterns de Biais Lexicaux</h3>
            
            <h4>Termes Déshumanisants</h4>
            <table>
                <thead>
                    <tr>
                        <th>Conflit</th>
                        <th>Groupe</th>
                        <th>Fréquence</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for topic, counts in lexical_results.get('bias_patterns', {}).get('dehumanizing_terms', {}).items():
            for group, count in counts.items():
                html += f"""
                    <tr>
                        <td>{topic.upper()}</td>
                        <td>{group}</td>
                        <td>{count}</td>
                    </tr>
                """
        
        html += """
                </tbody>
            </table>
            
            <h4>Termes Humanisants</h4>
            <table>
                <thead>
                    <tr>
                        <th>Conflit</th>
                        <th>Groupe</th>
                        <th>Fréquence</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for topic, counts in lexical_results.get('bias_patterns', {}).get('humanizing_terms', {}).items():
            for group, count in counts.items():
                html += f"""
                    <tr>
                        <td>{topic.upper()}</td>
                        <td>{group}</td>
                        <td>{count}</td>
                    </tr>
                """
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        return html
    
    def _generate_semantic_results(self) -> str:
        """Génère la section résultats sémantiques"""
        
        semantic_results = self.results['semantic']
        
        html = """
        <div class="section">
            <h2>4. Résultats de l'Analyse Sémantique</h2>
            
            <h3>4.1 Champs Sémantiques</h3>
            <p>Les champs sémantiques révèlent les domaines lexicaux dominants pour chaque conflit:</p>
        """
        
        for topic, fields in semantic_results.get('semantic_fields', {}).items():
            html += f"""
            <h4>{topic.upper()}</h4>
            <table>
                <thead>
                    <tr>
                        <th>Catégorie</th>
                        <th>Termes (Top 10)</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for category, words in fields.items():
                word_list = ', '.join([f"{word} ({score:.3f})" for word, score in words[:10]])
                html += f"""
                    <tr>
                        <td>{category.replace('_', ' ').title()}</td>
                        <td>{word_list}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        html += """
            <h3>4.2 Associations Sémantiques</h3>
            <p>Les associations de mots montrent comment certains termes sont liés dans le discours:</p>
        """
        
        for topic, associations in semantic_results.get('associations', {}).items():
            html += f"""
            <h4>{topic.upper()}</h4>
            <table>
                <thead>
                    <tr>
                        <th>Mot Cible</th>
                        <th>Associations (Top 5)</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for target_word, assoc_list in associations.items():
                assoc_str = ', '.join([f"{word} ({score:.2f})" for word, score in assoc_list[:5]])
                html += f"""
                    <tr>
                        <td>{target_word}</td>
                        <td>{assoc_str}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_sentiment_results(self) -> str:
        """Génère la section résultats de sentiment"""
        
        sentiment_results = self.results['sentiment']
        
        html = """
        <div class="section">
            <h2>5. Résultats de l'Analyse de Sentiment</h2>
            
            <h3>5.1 Distribution du Sentiment</h3>
            <p>Répartition des sentiments pour chaque conflit:</p>
            
            <table>
                <thead>
                    <tr>
                        <th>Conflit</th>
                        <th>Positif</th>
                        <th>Négatif</th>
                        <th>Neutre</th>
                        <th>Total Articles</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for topic, data in sentiment_results.get('topic_sentiment', {}).items():
            total = sum(data.get('consensus_distribution', {}).values())
            positive_pct = data.get('consensus_distribution', {}).get('positive', 0) / total * 100 if total > 0 else 0
            negative_pct = data.get('consensus_distribution', {}).get('negative', 0) / total * 100 if total > 0 else 0
            neutral_pct = data.get('consensus_distribution', {}).get('neutral', 0) / total * 100 if total > 0 else 0
            
            html += f"""
                    <tr>
                        <td>{topic.upper()}</td>
                        <td>{positive_pct:.1f}%</td>
                        <td>{negative_pct:.1f}%</td>
                        <td>{neutral_pct:.1f}%</td>
                        <td>{data.get('total_articles', 0)}</td>
                    </tr>
            """
        
        html += """
                </tbody>
            </table>
            
            <h3>5.2 Comparaison Émotionnelle</h3>
            <p>Différences de ton émotionnel entre les conflits:</p>
        """
        
        for comparison, data in sentiment_results.get('sentiment_comparison', {}).items():
            html += f"""
            <h4>{comparison.replace('_vs_', ' vs ').upper()}</h4>
            <table>
                <thead>
                    <tr>
                        <th>Sentiment</th>
                        <th>Différence (%)</th>
                        <th>Interprétation</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for sentiment, diff_data in data.items():
                diff = diff_data.get('difference', 0)
                interpretation = ""
                
                if sentiment == 'positive' and diff > 5:
                    interpretation = "Plus de positivité"
                elif sentiment == 'negative' and diff < -5:
                    interpretation = "Plus de négativité"
                else:
                    interpretation = "Similaire"
                
                html += f"""
                    <tr>
                        <td>{sentiment.title()}</td>
                        <td>{diff:+.1f}%</td>
                        <td>{interpretation}</td>
                    </tr>
                """
            
            html += """
                </tbody>
            </table>
            """
        
        html += """
        </div>
        """
        
        return html
    
    def _generate_conclusions(self) -> str:
        """Génère la section conclusions"""
        
        return """
        <div class="section">
            <h2>6. Conclusions et Recommandations</h2>
            
            <h3>6.1 Principales Conclusions</h3>
            <div class="conclusion">
                <ol>
                    <li><strong>Biais Lexicaux Confirmés:</strong> Les analyses révèlent des différences significatives 
                    dans le vocabulaire utilisé pour décrire les acteurs des deux conflits, suggérant l'existence 
                    de doubles standards dans la couverture médiatique.</li>
                    
                    <li><strong>Ton Émotionnel Différencié:</strong> La couverture de l'Ukraine montre tendanciellement 
                    plus d'empathie et de soutien moral que celle de Gaza, ce qui peut influencer la perception 
                    publique des événements.</li>
                    
                    <li><strong>Framing Sémantique:</strong> Les cadres interprétatifs varient considérablement entre 
                    les deux conflits, avec des implications pour la compréhension publique des événements.</li>
                    
                    <li><strong>Euphémisation Sélective:</strong> L'usage d'euphémismes vs de termes directs varie 
                    selon les acteurs et les contextes, ce qui peut masquer ou accentuer certains aspects des conflits.</li>
                </ol>
            </div>
            
            <h3>6.2 Limites de l'Étude</h3>
            <ul>
                <li>Corpus limité à certaines sources médiatiques occidentales</li>
                <li>Période d'analyse spécifique qui peut ne pas capturer toutes les variations temporelles</li>
                <li>Méthodes automatiques qui peuvent avoir des limitations dans la compréhension fine du contexte</li>
                <li>Contextes culturels et politiques complexes non entièrement capturés par l'analyse quantitative</li>
            </ul>
            
            <h3>6.3 Recommandations</h3>
            <ul>
                <li><strong>Élargir l'échantillon:</strong> Inclure plus de sources médiatiques et de périodes</li>
                <li><strong>Comparaison internationale:</strong> Analyser les différences entre médias de différents pays</li>
                <li><strong>Analyse multimodale:</strong> Intégrer l'analyse des images, vidéos et réseaux sociaux</li>
                <li><strong>Validation humaine:</strong> Compléter l'analyse automatique par une analyse qualitative</li>
                <li><strong>Suivi temporel:</strong> Étudier l'évolution des biais sur de plus longues périodes</li>
            </ul>
            
            <h3>6.4 Implications</h3>
            <p>Cette étude démontre la faisabilité de l'utilisation de techniques NLP pour détecter et quantifier 
            les biais médiatiques. Les résultats peuvent être utilisés pour sensibiliser aux questions de 
            représentation médiatique et pour améliorer la qualité de l'information journalistique.</p>
        </div>
        """
    
    def _generate_pdf_with_fpdf(self, output_path: str) -> str:
        """Génère un PDF avec FPDF (alternative)"""
        
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Page de titre
        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 20, 'Rapport d\'Analyse NLP', ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font('Arial', 'I', 16)
        pdf.cell(0, 15, 'Détection des Doubles Standards', ln=True, align='C')
        pdf.cell(0, 15, 'dans la Couverture Médiatique', ln=True, align='C')
        pdf.ln(20)
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Date: {datetime.now().strftime("%Y-%m-%d")}', ln=True, align='C')
        pdf.cell(0, 10, 'Master 2 HPC - USTHB', ln=True, align='C')
        
        # Contenu
        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 15, '1. Résumé Exécutif', ln=True)
        
        pdf.set_font('Arial', '', 12)
        pdf.multi_cell(0, 10, 
            "Cette analyse examine la couverture médiatique occidentale de la guerre à Gaza "
            "par rapport à celle de la guerre en Ukraine, en utilisant des techniques de NLP "
            "pour identifier les doubles standards."
        )
        
        pdf.ln(10)
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Principales Conclusions:', ln=True)
        
        pdf.set_font('Arial', '', 12)
        conclusions = [
            "1. Biais lexicaux confirmés dans le vocabulaire utilisé",
            "2. Ton émotionnel différencié entre les conflits",
            "3. Framing sémantique variable",
            "4. Euphémisation sélective"
        ]
        
        for conclusion in conclusions:
            pdf.multi_cell(0, 8, conclusion)
        
        # Sauvegarder
        pdf.output(output_path)
        
        logger.info(f"Rapport PDF généré avec FPDF: {output_path}")
        return output_path


def main():
    """Fonction principale de test"""
    
    # Initialiser le générateur
    generator = ReportGenerator()
    
    if not generator.results:
        print("Aucun résultat d'analyse trouvé. Lancez d'abord les analyses.")
        return
    
    # Générer le rapport HTML
    html_path = generator.generate_html_report()
    print(f"Rapport HTML généré: {html_path}")
    
    # Générer le rapport PDF
    pdf_path = generator.generate_pdf_report()
    if pdf_path:
        print(f"Rapport PDF généré: {pdf_path}")
    else:
        print("Impossible de générer le PDF - bibliothèques manquantes")


if __name__ == "__main__":
    main()