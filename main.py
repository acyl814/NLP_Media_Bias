"""
Script principal pour le projet NLP Bias Analyzer
Orchestre toutes les étapes du pipeline d'analyse
"""

import os
import sys
import argparse
import logging
from datetime import datetime

# Ajouter les répertoires au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestre le pipeline complet d'analyse NLP"""
    
    def __init__(self, config_path="config/config.yaml"):
        self.config_path = config_path
        self.results = {}
    
    def run_step(self, step_name: str, **kwargs):
        """
        Exécute une étape spécifique du pipeline
        
        Args:
            step_name: Nom de l'étape
            **kwargs: Arguments spécifiques à l'étape
        """
        
        logger.info(f"Exécution de l'étape: {step_name}")
        
        if step_name == "generate_sample_corpus":
            return self._generate_sample_corpus()
        
        elif step_name == "preprocess":
            return self._preprocess_corpus(kwargs.get('corpus_path'))
        
        elif step_name == "lexical_analysis":
            return self._run_lexical_analysis(kwargs.get('corpus_path'))
        
        elif step_name == "semantic_analysis":
            return self._run_semantic_analysis(kwargs.get('corpus_path'))
        
        elif step_name == "sentiment_analysis":
            return self._run_sentiment_analysis(kwargs.get('corpus_path'))
        
        elif step_name == "visualize":
            return self._generate_visualizations()
        
        elif step_name == "generate_report":
            return self._generate_report()
        
        elif step_name == "run_web_interface":
            return self._run_web_interface()
        
        else:
            raise ValueError(f"Éape inconnue: {step_name}")
    
    def _generate_sample_corpus(self):
        """Génère un corpus d'exemple"""
        
        try:
            from data_collection.sample_generator import SampleCorpusGenerator
            
            generator = SampleCorpusGenerator()
            corpus = generator.generate_corpus()
            
            logger.info(f"Corpus d'exemple généré: {len(corpus)} articles")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du corpus: {e}")
            return False
    
    def _preprocess_corpus(self, corpus_path=None):
        """Prétraite le corpus"""
        
        try:
            from preprocessing.text_processor import TextProcessor
            
            if not corpus_path:
                # Trouver le dernier corpus
                import glob
                corpus_files = glob.glob("corpus/corpus_*.json")
                if corpus_files:
                    corpus_path = max(corpus_files)
                else:
                    logger.error("Aucun fichier corpus trouvé")
                    return False
            
            processor = TextProcessor(use_spacy=False)  # NLTK plus rapide
            output_path = processor.process_corpus(corpus_path)
            
            logger.info(f"Corpus prétraité: {output_path}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du prétraitement: {e}")
            return False
    
    def _run_lexical_analysis(self, corpus_path=None):
        """Exécute l'analyse lexicale"""
        
        try:
            from analysis.lexical_analyzer import LexicalAnalyzer
            
            if not corpus_path:
                # Trouver le dernier corpus prétraité
                import glob
                preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
                if preprocessed_files:
                    corpus_path = max(preprocessed_files)
                else:
                    logger.error("Aucun corpus prétraité trouvé")
                    return False
            
            analyzer = LexicalAnalyzer()
            analyzer.load_corpus(corpus_path)
            results = analyzer.analyze_all()
            
            self.results['lexical'] = results
            logger.info("Analyse lexicale complétée")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse lexicale: {e}")
            return False
    
    def _run_semantic_analysis(self, corpus_path=None):
        """Exécute l'analyse sémantique"""
        
        try:
            from analysis.semantic_analyzer import SemanticAnalyzer
            
            if not corpus_path:
                # Trouver le dernier corpus prétraité
                import glob
                preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
                if preprocessed_files:
                    corpus_path = max(preprocessed_files)
                else:
                    logger.error("Aucun corpus prétraité trouvé")
                    return False
            
            analyzer = SemanticAnalyzer()
            analyzer.load_corpus(corpus_path)
            results = analyzer.analyze_all()
            
            self.results['semantic'] = results
            logger.info("Analyse sémantique complétée")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse sémantique: {e}")
            return False
    
    def _run_sentiment_analysis(self, corpus_path=None):
        """Exécute l'analyse de sentiment"""
        
        try:
            from analysis.sentiment_analyzer import SentimentAnalyzer
            
            if not corpus_path:
                # Trouver le dernier corpus prétraité
                import glob
                preprocessed_files = glob.glob("preprocessed/preprocessed_*.json")
                if preprocessed_files:
                    corpus_path = max(preprocessed_files)
                else:
                    logger.error("Aucun corpus prétraité trouvé")
                    return False
            
            analyzer = SentimentAnalyzer()
            analyzer.load_corpus(corpus_path)
            results = analyzer.analyze_all()
            
            self.results['sentiment'] = results
            logger.info("Analyse de sentiment complétée")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de l'analyse de sentiment: {e}")
            return False
    
    def _generate_visualizations(self):
        """Génère les visualisations"""
        
        try:
            from visualization.visualizer import Visualizer
            
            visualizer = Visualizer()
            visualizer.load_results("analysis_results")
            visualizer.generate_all_visualizations("visualizations")
            
            logger.info("Visualisations générées")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération des visualisations: {e}")
            return False
    
    def _generate_report(self):
        """Génère le rapport final"""
        
        try:
            from reports.report_generator import ReportGenerator
            
            generator = ReportGenerator()
            
            # Générer le rapport HTML
            html_path = generator.generate_html_report()
            
            # Générer le rapport PDF
            pdf_path = generator.generate_pdf_report()
            
            logger.info(f"Rapports générés: HTML={html_path}, PDF={pdf_path}")
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors de la génération du rapport: {e}")
            return False
    
    def _run_web_interface(self):
        """Lance l'interface web"""
        
        try:
            from web_interface.app import app
            
            # Configuration
            app.run(
                host="127.0.0.1",
                port=5000,
                debug=False,
                use_reloader=False
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Erreur lors du lancement de l'interface web: {e}")
            return False
    
    def run_full_pipeline(self):
        """Exécute le pipeline complet"""
        
        logger.info("="*60)
        logger.info("DÉMARRAGE DU PIPELINE COMPLET")
        logger.info("="*60)
        
        steps = [
            ("generate_sample_corpus", {}),
            ("preprocess", {}),
            ("lexical_analysis", {}),
            ("semantic_analysis", {}),
            ("sentiment_analysis", {}),
            ("visualize", {}),
            ("generate_report", {})
        ]
        
        results = {}
        
        for step_name, kwargs in steps:
            logger.info(f"\n{'-'*60}")
            logger.info(f"ÉTAPE: {step_name.upper()}")
            logger.info(f"{'-'*60}")
            
            success = self.run_step(step_name, **kwargs)
            results[step_name] = success
            
            if not success:
                logger.error(f"Étape {step_name} échouée!")
                break
            else:
                logger.info(f"Étape {step_name} réussie!")
        
        # Résumé
        logger.info("\n" + "="*60)
        logger.info("RÉSUMÉ DU PIPELINE")
        logger.info("="*60)
        
        for step_name, success in results.items():
            status = "✓" if success else "✗"
            logger.info(f"{status} {step_name}")
        
        successful_steps = sum(results.values())
        total_steps = len(results)
        
        logger.info(f"\nÉtapes réussies: {successful_steps}/{total_steps}")
        
        if successful_steps == total_steps:
            logger.info("\n🎉 PIPELINE TERMINÉ AVEC SUCCÈS!")
            logger.info("\nProchaines étapes:")
            logger.info("1. Lancez l'interface web: python main.py --web")
            logger.info("2. Ouvrez votre navigateur: http://127.0.0.1:5000")
            logger.info("3. Explorez les résultats!")
        else:
            logger.warning(f"\n⚠️  PIPELINE TERMINÉ AVEC {total_steps - successful_steps} ÉCHEC(S)")
        
        return results


def main():
    """Fonction principale"""
    
    parser = argparse.ArgumentParser(
        description="NLP Bias Analyzer - Détection des doubles standards dans la couverture médiatique"
    )
    
    parser.add_argument(
        "--step",
        choices=[
            "generate_sample_corpus",
            "preprocess",
            "lexical_analysis",
            "semantic_analysis",
            "sentiment_analysis",
            "visualize",
            "generate_report",
            "run_web_interface"
        ],
        help="Exécuter une étape spécifique du pipeline"
    )
    
    parser.add_argument(
        "--full",
        action="store_true",
        help="Exécuter le pipeline complet"
    )
    
    parser.add_argument(
        "--web",
        action="store_true",
        help="Lancer uniquement l'interface web"
    )
    
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Chemin vers le fichier de configuration"
    )
    
    args = parser.parse_args()
    
    # Initialiser l'orchestrateur
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    # Exécuter l'action demandée
    if args.full:
        # Pipeline complet
        orchestrator.run_full_pipeline()
    
    elif args.web:
        # Interface web seulement
        logger.info("Lancement de l'interface web...")
        orchestrator.run_step("run_web_interface")
    
    elif args.step:
        # Étape spécifique
        logger.info(f"Exécution de l'étape: {args.step}")
        success = orchestrator.run_step(args.step)
        
        if success:
            logger.info(f"✓ Étape {args.step} réussie!")
        else:
            logger.error(f"✗ Étape {args.step} échouée!")
            sys.exit(1)
    
    else:
        # Afficher l'aide
        parser.print_help()
        print("\nExemples d'utilisation:")
        print("  python main.py --full                    # Pipeline complet")
        print("  python main.py --web                     # Interface web seulement")
        print("  python main.py --step preprocess         # Prétraitement seulement")
        print("  python main.py --step lexical_analysis   # Analyse lexicale")


if __name__ == "__main__":
    main()