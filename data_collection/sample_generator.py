"""
Générateur de corpus d'exemple pour les tests
Crée des articles simulés pour Gaza et Ukraine avec des patterns de biais
"""

import json  # IMPORTANT: Doit être au niveau du module
import random
from datetime import datetime, timedelta
import os


class SampleCorpusGenerator:
    """Génère un corpus d'exemple avec des patterns de biais connus"""
    
    def __init__(self):
        self.sources = ["CNN", "BBC", "New York Times", "Reuters", "Guardian"]
        
        # Templates pour les articles de Gaza (avec biais)
        self.gaza_templates = [
            {
                "title_templates": [
                    "Israel Launches Military Operation in Gaza",
                    "Gaza Conflict Escalates as Violence Continues",
                    "Hamas Militants Fire Rockets at Israel",
                    "Israeli Forces Target Terrorist Infrastructure",
                    "Civilian Casualties Reported in Gaza Strikes"
                ],
                "content_templates": [
                    "The Israeli military conducted a series of targeted strikes against Hamas positions in Gaza. The operation targeted terrorist infrastructure and weapons storage facilities. Military officials reported that militants were using civilian areas as cover. The escalation began when Hamas fighters launched rockets into Israeli territory.",
                    
                    "Violence flared up again in the ongoing conflict as Israeli forces responded to attacks from Gaza. The military operation aimed to neutralize threats from Hamas militants. According to sources, the extremist group has been using civilian populations as human shields. The Israeli Defense Forces stated they are taking precautions to minimize civilian casualties.",
                    
                    "Tensions escalated following rocket attacks from Gaza, prompting Israeli military response. The conflict has resulted in significant damage to infrastructure. Hamas, the militant organization controlling Gaza, has been accused of launching attacks from residential areas. International observers called for restraint from both sides.",
                    
                    "Israeli forces launched a military operation in response to continued aggression from Gaza. The targeted strikes focused on Hamas military compounds and weapons depots. Military analysts noted that militants often operate from within civilian neighborhoods. The operation is part of ongoing efforts to ensure Israeli security."
                ]
            }
        ]
        
        # Templates pour les articles d'Ukraine (plus empathiques)
        self.ukraine_templates = [
            {
                "title_templates": [
                    "Ukrainian Forces Resist Russian Invasion",
                    "Brave Ukrainian Civilians Defend Their Homeland",
                    "Russia Commits War Crimes in Ukraine",
                    "Ukrainian Heroes Fight for Freedom",
                    "International Community Condemns Russian Aggression"
                ],
                "content_templates": [
                    "Ukrainian forces demonstrated remarkable resilience in their heroic resistance against the Russian invasion. Brave civilians have taken up arms to defend their homes and families. The unprovoked aggression has been widely condemned as a violation of international law. Ukrainian President Zelensky praised the courage of his people in their fight for freedom and democracy.",
                    
                    "The Ukrainian people continue to inspire the world with their courageous defense of their homeland. Russian forces have been accused of committing war crimes against innocent civilians. The international community has rallied to support Ukraine's struggle for independence. Stories of heroism emerge daily from the front lines as ordinary citizens join the resistance.",
                    
                    "Ukraine's heroic resistance against Russian aggression has captured global attention. Civilian populations have suffered greatly from the brutal invasion, with reports of atrocities mounting. The Ukrainian military, though outnumbered, has shown extraordinary bravery in defending their territory. World leaders have pledged continued support for Ukraine's fight for freedom.",
                    
                    "The Ukrainian spirit remains unbroken despite continued Russian attacks on civilian infrastructure. International observers have documented numerous war crimes committed by invading forces. Ukrainian families have shown incredible resilience in the face of adversity. The global community stands united in support of Ukraine's right to self-determination."
                ]
            }
        ]
        
        # Termes déshumanisants pour Gaza
        self.gaza_dehumanizing = ["militants", "terrorists", "fighters", "extremists", "gunmen", "Hamas"]
        
        # Termes humanisants pour Ukraine
        self.ukraine_humanizing = ["civilians", "victims", "families", "heroes", "brave", "resilient"]
    
    def generate_corpus(self, output_dir="corpus"):
        """Génère le corpus complet"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        all_articles = []
        
        # Générer des articles sur Gaza (50 articles)
        print("Génération des articles sur Gaza...")
        gaza_articles = self._generate_articles("gaza", 50)
        all_articles.extend(gaza_articles)
        
        # Générer des articles sur l'Ukraine (30 articles)
        print("Génération des articles sur l'Ukraine...")
        ukraine_articles = self._generate_articles("ukraine", 30)
        all_articles.extend(ukraine_articles)
        
        # Sauvegarder
        self._save_articles(all_articles, output_dir)
        
        print(f"Corpus généré: {len(all_articles)} articles")
        return all_articles
    
    def _generate_articles(self, topic, count):
        """Génère des articles pour un sujet donné"""
        
        articles = []
        
        if topic == "gaza":
            templates = self.gaza_templates[0]
        else:
            templates = self.ukraine_templates[0]
        
        for i in range(count):
            article = self._generate_single_article(topic, templates, i)
            articles.append(article)
        
        return articles
    
    def _generate_single_article(self, topic, templates, index):
        """Génère un article unique"""
        
        # Choisir un template aléatoire
        title_template = random.choice(templates["title_templates"])
        content_template = random.choice(templates["content_templates"])
        
        # Ajouter des variations
        title = self._vary_text(title_template, index)
        content = self._vary_text(content_template, index)
        
        # Ajouter des détails spécifiques selon le topic
        if topic == "gaza":
            content = self._add_gaza_specifics(content, index)
        else:
            content = self._add_ukraine_specifics(content, index)
        
        # Générer les métadonnées
        article = {
            "id": f"{topic}_{index+1:03d}",
            "url": f"https://example-news.com/{topic}/article-{index+1}",
            "title": title,
            "content": content,
            "topic": topic,
            "source": random.choice(self.sources),
            "authors": [f"Journalist {random.randint(1, 50)}"],
            "publish_date": self._generate_date(topic, index),
            "extraction_date": "2024-01-15T10:00:00",
            "word_count": len(content.split())
        }
        
        return article
    
    def _vary_text(self, text, index):
        """Ajoute des variations au texte"""
        
        variations = [
            "",
            " - Analysis",
            " - Report",
            ": What We Know",
            " - Latest Updates",
            " - Breaking News"
        ]
        
        return text + random.choice(variations)
    
    def _add_gaza_specifics(self, content, index):
        """Ajoute des détails spécifiques à Gaza"""
        
        specifics = [
            " The military operation lasted several hours.",
            " Palestinian officials reported casualties.",
            " Israeli officials stated the operation was successful.",
            " The conflict has drawn international attention.",
            " Humanitarian organizations expressed concern."
        ]
        
        return content + random.choice(specifics)
    
    def _add_ukraine_specifics(self, content, index):
        """Ajoute des détails spécifiques à l'Ukraine"""
        
        specifics = [
            " President Zelensky addressed the nation.",
            " The international community pledged support.",
            " Russian officials denied the allegations.",
            " NATO allies provided humanitarian aid.",
            " The UN Security Council held an emergency session."
        ]
        
        return content + random.choice(specifics)
    
    def _generate_date(self, topic, index):
        """Génère une date de publication"""
        
        if topic == "gaza":
            # Gaza: octobre 2023 à décembre 2024
            start_date = datetime(2023, 10, 7)
            end_date = datetime(2024, 12, 31)
        else:
            # Ukraine: février 2022 à décembre 2024
            start_date = datetime(2022, 2, 24)
            end_date = datetime(2024, 12, 31)
        
        # Calculer une date aléatoire dans la plage
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        
        random_date = start_date + timedelta(days=random_days)
        
        return random_date.isoformat()
    
    def _save_articles(self, articles, output_dir):
        """Sauvegarde les articles avec gestion des types pandas/NumPy"""
        
        import pandas as pd
        import numpy as np
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON - Les articles sont déjà en types Python natifs, pas besoin de conversion
        json_path = os.path.join(output_dir, f"corpus_{timestamp}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, indent=2, ensure_ascii=False)
        
        # CSV
        df = pd.DataFrame(articles)
        csv_path = os.path.join(output_dir, f"corpus_{timestamp}.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # Statistiques - Conversion MANUELLE en types Python natifs
        # Éviter d'utiliser .to_dict() sur les Series pandas
        topics_counts = df['topic'].value_counts()
        topics_dict = {str(k): int(v) for k, v in topics_counts.items()}
        
        sources_counts = df['source'].value_counts()
        sources_dict = {str(k): int(v) for k, v in sources_counts.items()}
        
        stats = {
            'total_articles': int(len(articles)),
            'topics': topics_dict,
            'sources': sources_dict,
            'total_words': int(df['word_count'].sum()),
            'avg_words': float(df['word_count'].mean()),
            'date_range': {
                'earliest': str(df['publish_date'].min()),
                'latest': str(df['publish_date'].max())
            }
        }
        
        stats_path = os.path.join(output_dir, f"stats_{timestamp}.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"Articles sauvegardés dans: {json_path}")
        print(f"Statistiques sauvegardées dans: {stats_path}")
        
        # Afficher les statistiques
        print("\n" + "="*60)
        print("STATISTIQUES DU CORPUS")
        print("="*60)
        print(f"Total d'articles: {stats['total_articles']}")
        print("\nRépartition par sujet:")
        for topic, count in stats['topics'].items():
            print(f"  - {topic}: {count} articles")
        print("\nRépartition par source:")
        for source, count in stats['sources'].items():
            print(f"  - {source}: {count} articles")
        print(f"\nTotal de mots: {stats['total_words']:,}")
        print(f"Mots moyens par article: {stats['avg_words']:.1f}")


def main():
    """Fonction principale"""
    
    generator = SampleCorpusGenerator()
    generator.generate_corpus()


if __name__ == "__main__":
    main()