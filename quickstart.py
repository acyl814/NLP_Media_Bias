"""
Script de démarrage rapide pour le projet NLP Bias Analyzer
Génère un corpus d'exemple et lance l'analyse complète en une seule commande
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_command(command, description):
    """Exécute une commande avec affichage du progrès"""
    
    logger.info(f"{'='*60}")
    logger.info(f"ÉTAPE: {description}")
    logger.info(f"COMMANDE: {command}")
    logger.info(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"✓ {description} réussie")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} échouée")
        logger.error(f"Erreur: {e}")
        return False


def quick_start():
    """Lance le pipeline complet en mode démonstration"""
    
    logger.info("="*60)
    logger.info("NLP BIAS ANALYZER - DÉMARRAGE RAPIDE")
    logger.info("="*60)
    
    logger.info("\nCe script va:")
    logger.info("1. Générer un corpus d'exemple (80 articles)")
    logger.info("2. Prétraiter les données")
    logger.info("3. Exécuter toutes les analyses")
    logger.info("4. Générer les visualisations")
    logger.info("5. Créer le rapport final")
    logger.info("6. Lancer l'interface web\n")
    
    # Vérifier que les dépendances sont installées
    logger.info("Vérification des dépendances...")
    
    # Liste des packages requis - CORRECTION ICI : utiliser les noms d'importation Python
    required_packages = [
        'nltk', 'pandas', 'numpy', 'matplotlib', 'seaborn',
        'plotly', 'sklearn', 'yaml', 'flask'  # Changé 'scikit-learn'→'sklearn', 'pyyaml'→'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Packages manquants: {', '.join(missing_packages)}")
        logger.info("\nInstallation des dépendances:")
        logger.info("pip install -r requirements.txt")
        return False
    
    # Créer les répertoires nécessaires
    directories = [
        'corpus', 'preprocessed', 'analysis_results',
        'visualizations', 'reports'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"✓ Répertoire créé/vérifié: {directory}")
    
    # Étape 1: Générer le corpus d'exemple - CORRIGÉ (guillemets doubles)
    if not run_command(
        'python -c "from data_collection.sample_generator import SampleCorpusGenerator; SampleCorpusGenerator().generate_corpus()"',
        "Génération du corpus d'exemple"
    ):
        return False
    
    # Étape 2: Prétraitement - CORRIGÉ (guillemets doubles)
    if not run_command(
    'python -c "from preprocessing.text_processor import TextProcessor; TextProcessor(use_spacy=False).process_corpus(\\"corpus/corpus_20251223_214030.json\\")"',
    "Prétraitement des données"
    ):
       return False
    
    # Étape 3: Analyses
    analyses = [
        ("lexical_analysis", "Analyse lexicale"),
        ("semantic_analysis", "Analyse sémantique"),
        ("sentiment_analysis", "Analyse de sentiment")
    ]
    
    for analysis_name, description in analyses:
        if not run_command(
            f'python -c "from main import PipelineOrchestrator; PipelineOrchestrator().run_step(\\"{analysis_name}\\")"',
            description
        ):
            return False
    
    # Étape 4: Visualisations - CORRIGÉ (guillemets doubles)
    if not run_command(
        'python -c "from visualization.visualizer import Visualizer; Visualizer().generate_all_visualizations()"',
        "Génération des visualisations"
    ):
        return False
    
    # Étape 5: Rapport - CORRIGÉ (guillemets doubles)
    if not run_command(
        'python -c "from reports.report_generator import ReportGenerator; ReportGenerator().generate_html_report()"',
        "Génération du rapport"
    ):
        return False
    
    # Résumé final
    logger.info("\n" + "="*60)
    logger.info("PIPELINE TERMINÉ AVEC SUCCÈS!")
    logger.info("="*60)
    
    logger.info("\nFichiers générés:")
    
    # Lister les fichiers créés
    for directory in directories:
        if os.path.exists(directory):
            files = os.listdir(directory)
            if files:
                logger.info(f"\n{directory}/:")
                for file in sorted(files)[-3:]:  # 3 derniers fichiers
                    logger.info(f"  - {file}")
    
    logger.info("\n" + "="*60)
    logger.info("PROCHAINES ÉTAPES:")
    logger.info("="*60)
    logger.info("1. Lancez l'interface web:")
    logger.info("   python main.py --web")
    logger.info("")
    logger.info("2. Ouvrez votre navigateur:")
    logger.info("   http://127.0.0.1:5000")
    logger.info("")
    logger.info("3. Explorez les résultats!")
    logger.info("\n" + "="*60)
    
    # Demander si l'utilisateur veut lancer l'interface web
    try:
        response = input("\nVoulez-vous lancer l'interface web maintenant? (y/n): ")
        if response.lower() == 'y':
            logger.info("Lancement de l'interface web...")
            subprocess.run("python main.py --web", shell=True)
    except EOFError:
        # Gestion du cas où il n'y a pas d'entrée interactive
        logger.info("\nMode non-interactif détecté. Interface web non lancée.")
    
    return True


def demo_mode():
    """Mode démonstration avec sortie formatée"""
    
    print("\n" + "="*80)
    print("NLP BIAS ANALYZER - DÉMONSTRATION")
    print("="*80)
    print()
    print("Ce script exécute une démonstration complète du pipeline d'analyse.")
    print("Il utilisera des données générées automatiquement pour illustrer")
    print("les fonctionnalités du système.")
    print()
    print("Durée estimée: 5-10 minutes")
    print()
    
    input("Appuyez sur Entrée pour continuer...")
    
    success = quick_start()
    
    if not success:
        print("\n❌ La démonstration a échoué.")
        print("Vérifiez les logs ci-dessus pour plus d'informations.")
        return False
    
    print("\n" + "="*80)
    print("🎉 DÉMONSTRATION TERMINÉE AVEC SUCCÈS!")
    print("="*80)
    
    return True


def show_help():
    """Affiche l'aide détaillée"""
    
    help_text = """
NLP Bias Analyzer - Aide

COMMANDES:

1. Démarrage rapide complet:
   python quickstart.py

2. Mode démonstration:
   python quickstart.py --demo

3. Pipeline complet:
   python main.py --full

4. Interface web uniquement:
   python main.py --web

5. Étape spécifique:
   python main.py --step <nom_etape>

ÉTAPES DISPONIBLES:
- generate_sample_corpus : Génère un corpus d'exemple
- preprocess            : Prétraite les données
- lexical_analysis      : Analyse lexicale
- semantic_analysis     : Analyse sémantique
- sentiment_analysis    : Analyse de sentiment
- visualize            : Génère les visualisations
- generate_report      : Crée le rapport final
- run_web_interface    : Lance l'interface web

EXEMPLES:

# Pipeline complet avec configuration personnalisée
python main.py --full --config config/my_config.yaml

# Analyse lexicale seulement
python main.py --step lexical_analysis

# Lancer l'interface web
python main.py --web

# Mode démonstration
python quickstart.py --demo

FICHIERS DE SORTIE:

- corpus/corpus_*.json          : Articles collectés
- preprocessed/preprocessed_*.json : Données nettoyées
- analysis_results/*.json       : Résultats d'analyse
- visualizations/*.html         : Graphiques interactifs
- visualizations/*.png          : Graphiques statiques
- reports/report.html           : Rapport HTML
- reports/report.pdf            : Rapport PDF

INTERFACE WEB:

Une fois lancée, l'interface est accessible à:
http://127.0.0.1:5000

Pages disponibles:
- /              : Accueil
- /corpus        : Explorateur de corpus
- /analysis      : Tableau de bord d'analyse
- /bias-detector : Détecteur de biais
- /visualizations: Galerie de visualisations
- /report        : Rapport complet
"""
    
    print(help_text)


def main():
    """Fonction principale"""
    
    import sys
    
    # Vérifier les arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h', 'help']:
            show_help()
            return
        
        if sys.argv[1] == '--demo':
            demo_mode()
            return
    
    # Mode par défaut: démarrage rapide
    quick_start()


if __name__ == "__main__":
    main()