# NLP Bias Analyzer

**Détection des doubles standards dans la couverture médiatique de la guerre à Gaza**

Projet de fin d'études - Master 2 HPC  
Université des Sciences et de la Technologie Houari Boumediene (USTHB)

---

## 📋 Table des matières

- [Description](#description)
- [Fonctionnalités](#fonctionnalités)
- [Architecture](#architecture)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Résultats attendus](#résultats-attendus)
- [Contributeurs](#contributeurs)

---

## 🎯 Description

Ce projet utilise des techniques de **Traitement Automatique du Langage Naturel (NLP)** pour analyser et quantifier les biais dans la couverture médiatique occidentale de la guerre à Gaza, en la comparant à celle de la guerre en Ukraine.

### Objectifs principaux

- **Analyser les biais lexicaux**: Identifier les différences de vocabulaire utilisé pour décrire les acteurs des deux conflits
- **Examiner le ton émotionnel**: Comparer l'empathie et l'humanisation dans les descriptions
- **Étudier le framing sémantique**: Analyser les cadres interprétatifs et les associations de mots
- **Détecter les euphémismes**: Identifier l'usage sélectif de termes techniques vs directs

### Hypothèses de recherche

1. **Biais internes**: Les médias occidentaux appliquent des standards différents aux acteurs palestiniens et israéliens
2. **Biais systémiques**: La couverture de Gaza montre moins d'empathie que celle de l'Ukraine

---

## ✨ Fonctionnalités

### 🔍 Analyses

- **Analyse Lexicale**
  - Fréquences de mots
  - Analyse TF-IDF
  - Cooccurrences et associations
  - N-grams (bigrams, trigrams)
  - Détection de patterns de biais (termes déshumanisants vs humanisants)

- **Analyse Sémantique**
  - Concordance (contextes d'utilisation)
  - Champs sémantiques
  - Associations entre mots
  - Collocations
  - Comparaison des contextes entre conflits

- **Analyse de Sentiment**
  - Analyse multi-modèles (TextBlob, VADER, Transformers)
  - Sentiment par article et par topic
  - Analyse par mots-cibles
  - Comparaison émotionnelle
  - Évolution temporelle

### 📊 Visualisations

- Graphiques de fréquences de mots
- Réseaux de cooccurrences
- Cartes de chaleur de contextes
- Distribution du sentiment
- Tableaux de bord comparatifs

### 🌐 Interface Web

- Explorateur de corpus
- Tableau de bord d'analyse
- Détecteur de biais interactif
- Galerie de visualisations
- Génération de rapports PDF

### 📄 Rapports

- Rapport HTML interactif
- Rapport PDF professionnel
- Export CSV des résultats
- Visualisations intégrées

---

## 🏗 Architecture

```
nlp_bias_analysis/
│
├── config/
│   └── config.yaml              # Configuration du projet
│
├── data_collection/
│   ├── collectors.py            # Collecteurs d'articles
│   └── sample_generator.py      # Générateur de corpus d'exemple
│
├── preprocessing/
│   └── text_processor.py        # Nettoyage et prétraitement
│
├── analysis/
│   ├── lexical_analyzer.py      # Analyse lexicale
│   ├── semantic_analyzer.py     # Analyse sémantique
│   └── sentiment_analyzer.py    # Analyse de sentiment
│
├── visualization/
│   └── visualizer.py            # Génération de visualisations
│
├── web_interface/
│   ├── app.py                   # Application Flask
│   └── templates/               # Templates HTML
│
├── reports/
│   └── report_generator.py      # Génération de rapports
│
├── corpus/                      # Données collectées
├── preprocessed/                # Données prétraitées
├── analysis_results/            # Résultats d'analyse
├── visualizations/              # Graphiques et visualisations
│
├── main.py                      # Script principal
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation
```

---

## 🚀 Installation

### Prérequis

- Python 3.8 ou plus récent
- pip (gestionnaire de packages Python)

### Étapes d'installation

1. **Cloner le projet**
   ```bash
   git clone <url-du-repo>
   cd nlp_bias_analysis
   ```

2. **Créer un environnement virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # ou
   venv\Scripts\activate     # Windows
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

4. **Télécharger les ressources NLTK**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   nltk.download('averaged_perceptron_tagger')
   ```

5. **Installer spaCy (optionnel)**
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. **Installer WeasyPrint pour les PDFs (optionnel)**
   ```bash
   pip install weasyprint
   ```

---

## 💻 Utilisation

### Pipeline complet

Exécutez toutes les étapes du pipeline d'analyse:

```bash
python main.py --full
```

### Étapes individuelles

```bash
# 1. Générer un corpus d'exemple
python main.py --step generate_sample_corpus

# 2. Prétraiter les données
python main.py --step preprocess

# 3. Analyse lexicale
python main.py --step lexical_analysis

# 4. Analyse sémantique
python main.py --step semantic_analysis

# 5. Analyse de sentiment
python main.py --step sentiment_analysis

# 6. Générer les visualisations
python main.py --step visualize

# 7. Générer le rapport
python main.py --step generate_report
```

### Interface web

```bash
# Lancer l'interface Flask
python main.py --web
```

Puis ouvrez votre navigateur: `http://127.0.0.1:5000`

### Utilisation avancée

```bash
# Spécifier un fichier de configuration personnalisé
python main.py --full --config config/my_config.yaml

# Exécuter avec des logs détaillés
python main.py --full --log-level DEBUG
```

---

## 📖 Structure du projet

### Configuration (`config/`)

- `config.yaml`: Paramètres de collecte, analyse et visualisation

### Collecte de données (`data_collection/`)

- **collectors.py**: Collecteurs pour CNN, BBC, New York Times
- **sample_generator.py**: Génère un corpus d'exemple avec des patterns de biais connus

### Prétraitement (`preprocessing/`)

- **text_processor.py**: Nettoyage, tokenisation, lemmatisation

### Analyse (`analysis/`)

- **lexical_analyzer.py**: Analyse des fréquences, TF-IDF, patterns de biais
- **semantic_analyzer.py**: Concordance, champs sémantiques, associations
- **sentiment_analyzer.py**: Analyse multi-modèles de sentiment

### Visualisation (`visualization/`)

- **visualizer.py**: Génération de graphiques interactifs et statiques

### Interface web (`web_interface/`)

- **app.py**: Application Flask
- **templates/**: Pages HTML (accueil, corpus, analyse, etc.)

### Rapports (`reports/`)

- **report_generator.py**: Génération de rapports PDF et HTML

---

## 📊 Résultats attendus

### Analyse 1: Biais internes

**Hypothèse**: Les médias occidentaux appliquent des standards différents aux acteurs palestiniens et israéliens.

**Observations attendues**:
- Les Palestiniens sont décrits avec des termes déshumanisants ("militants", "terrorists")
- Les Israéliens sont décrits avec empathie ("civilians", "victims")
- Attribution de responsabilité asymétrique
- Contextualisation différenciée

### Analyse 2: Biais systémiques

**Hypothèse**: La couverture de Gaza montre moins d'empathie que celle de l'Ukraine.

**Observations attendues**:
- Pour l'Ukraine: ton héroïque et empathique ("heroic resistance", "fight for freedom")
- Pour Gaza: ton neutre et technique ("conflict", "military operation")
- Euphémisation plus marquée pour Gaza
- Humanisation limitée, privilégiant les statistiques aux récits personnels

### Visualisations clés

1. **Fréquences de mots**: Top 50 mots les plus fréquents par conflit
2. **Réseaux de cooccurrences**: Mots associés à "Palestinians" vs "Ukrainians"
3. **Distribution du sentiment**: Comparaison des tons émotionnels
4. **Cartes de chaleur**: Contextes d'utilisation des mots-clés
5. **Tableaux de bord**: Synthèse des biais détectés

---

## 🎓 Évaluation

### Critères de réussite

- **Collecte de données**: 50-100 articles pour Gaza, 30-50 pour l'Ukraine
- **Analyse linguistique**: Patterns pertinents et argumentés
- **Visualisations**: Graphiques et statistiques obligatoires
- **Originalité**: Méthodologie et visualisations innovantes

### Livrables

1. **Rapport PDF** (20 pages max)
   - Méthodologie et résultats
   - Visualisations et analyse critique
   - Déclaration de contribution

2. **Dépôt GitHub**
   - Corpus organisé
   - Code source complet
   - Scripts de reproductibilité

3. **Application/Démonstration**
   - Interface utilisateur
   - Visualisations interactives
   - Consultation des corpus

---

## 👥 Contributeurs

Ce projet a été développé dans le cadre du module **Natural Language Processing** du **Master 2 HPC** à l'**USTHB**.

**Instructeur**: Dr. S. KALI ALI (skaliali.usthb@gmail.com)

### Guide de contribution

1. Fork le projet
2. Créez une branche pour votre fonctionnalité (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrez une Pull Request

---

## 📄 Licence

Ce projet est destiné à des fins éducatives dans le cadre du Master 2 HPC à l'USTHB.

---

## 🙏 Remerciements

- **Dr. S. KALI ALI** - Instructeur du module NLP
- **USTHB** - Université des Sciences et de la Technologie Houari Boumediene
- **Communauté open source** - Pour les excellentes bibliothèques NLP utilisées

---

## 📞 Support

Pour toute question ou suggestion, veuillez contacter:

- Email: skaliali.usthb@gmail.com
- Module: Natural Language Processing - Master 2 HPC
- Année universitaire: 2025-2026

---

<div align="center">
    <p><strong>NLP Bias Analyzer</strong></p>
    <p>Détection des doubles standards dans la couverture médiatique</p>
    <p><em>Master 2 HPC - USTHB - 2025-2026</em></p>
</div>