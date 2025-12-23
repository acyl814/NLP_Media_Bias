# 🔍 Projet NLP - Detection des Biais Mediatiques

## 🚀 Installation

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
python -m spacy download en_core_web_md
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 💻 Utilisation

1. Ajoutez vos URLs dans:
   - backend/data/raw/gaza_urls.txt
   - backend/data/raw/ukraine_urls.txt

2. Lancez le backend:
   ```bash
   cd backend
   venv\Scripts\activate
   python main.py --auto
   ```

3. Lancez le frontend:
   ```bash
   cd frontend
   npm run dev
   ```

Ouvrez http://localhost:3000

## 📁 Structure

- backend/  : Scripts Python NLP
- frontend/ : Interface Next.js
- docs/     : Documentation
