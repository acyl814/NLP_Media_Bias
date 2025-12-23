@echo off
chcp 65001 >nul
setlocal EnableDelayedExpansion

echo ================================================================================
echo 🚀 INSTALLATION PROJET NLP - DETECTION DES BIAIS MEDIATIQUES
echo    Generation automatique de tous les fichiers...
echo ================================================================================
echo.

set PROJECT_NAME=nlp-media-bias

:: Créer structure
echo [ETAPE 1] Creation de la structure...
mkdir %PROJECT_NAME%\backend\data\raw 2>nul
mkdir %PROJECT_NAME%\backend\data\processed 2>nul
mkdir %PROJECT_NAME%\backend\data\results\lexical 2>nul
mkdir %PROJECT_NAME%\backend\data\results\semantic 2>nul
mkdir %PROJECT_NAME%\backend\data\results\sentiment 2>nul
mkdir %PROJECT_NAME%\backend\scripts 2>nul
mkdir %PROJECT_NAME%\backend\config 2>nul
mkdir %PROJECT_NAME%\backend\models 2>nul
mkdir %PROJECT_NAME%\frontend\app\about 2>nul
mkdir %PROJECT_NAME%\frontend\app\corpus 2>nul
mkdir %PROJECT_NAME%\frontend\app\analysis\lexical 2>nul
mkdir %PROJECT_NAME%\frontend\app\analysis\semantic 2>nul
mkdir %PROJECT_NAME%\frontend\app\analysis\sentiment 2>nul
mkdir %PROJECT_NAME%\frontend\components\layout 2>nul
mkdir %PROJECT_NAME%\frontend\components\ui 2>nul
mkdir %PROJECT_NAME%\frontend\components\charts 2>nul
mkdir %PROJECT_NAME%\frontend\public\data 2>nul
mkdir %PROJECT_NAME%\frontend\lib 2>nul
mkdir %PROJECT_NAME%\docs 2>nul

cd %PROJECT_NAME%

echo ✅ Structure creee
echo.

:: README
echo [ETAPE 2] Creation des fichiers...
(
echo # 🔍 Projet NLP - Detection des Biais Mediatiques
echo.
echo ## 🚀 Installation
echo.
echo ### Backend
echo ```bash
echo cd backend
echo python -m venv venv
echo venv\Scripts\activate
echo pip install -r requirements.txt
echo python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
echo python -m spacy download en_core_web_md
echo ```
echo.
echo ### Frontend
echo ```bash
echo cd frontend
echo npm install
echo npm run dev
echo ```
echo.
echo ## 💻 Utilisation
echo.
echo 1. Ajoutez vos URLs dans:
echo    - backend/data/raw/gaza_urls.txt
echo    - backend/data/raw/ukraine_urls.txt
echo.
echo 2. Lancez le backend:
echo    ```bash
echo    cd backend
echo    venv\Scripts\activate
echo    python main.py --auto
echo    ```
echo.
echo 3. Lancez le frontend:
echo    ```bash
echo    cd frontend
echo    npm run dev
echo    ```
echo.
echo Ouvrez http://localhost:3000
echo.
echo ## 📁 Structure
echo.
echo - backend/  : Scripts Python NLP
echo - frontend/ : Interface Next.js
echo - docs/     : Documentation
) > README.md

:: URLs templates
(
echo # URLs des articles Gaza ^(50-100 articles^)
echo # Une URL par ligne
echo.
echo # Exemples:
echo # https://www.cnn.com/2024/10/15/middleeast/gaza-strikes/index.html
echo # https://www.bbc.com/news/world-middle-east-67123456
echo.
echo # Ajoutez vos URLs ici
) > backend\data\raw\gaza_urls.txt

(
echo # URLs des articles Ukraine ^(30-50 articles^)
echo # Une URL par ligne
echo.
echo # Exemples:
echo # https://www.cnn.com/2024/02/20/europe/ukraine-war/index.html
echo.
echo # Ajoutez vos URLs ici
) > backend\data\raw\ukraine_urls.txt

echo ✅ Fichiers de base crees
echo.

:: Guide Windows
(
echo ================================================================================
echo 📖 GUIDE D'UTILISATION WINDOWS
echo ================================================================================
echo.
echo ETAPE 1: Installer Python et Node.js
echo   - Python 3.8+: https://www.python.org/downloads/
echo   - Node.js 18+: https://nodejs.org/
echo.
echo ETAPE 2: Installer les dependances
echo.
echo   Backend:
echo     cd backend
echo     python -m venv venv
echo     venv\Scripts\activate
echo     pip install -r requirements.txt
echo     python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
echo     python -m spacy download en_core_web_md
echo.
echo   Frontend:
echo     cd frontend
echo     npm install
echo.
echo ETAPE 3: Ajouter vos URLs
echo   - Editez: backend\data\raw\gaza_urls.txt
echo   - Editez: backend\data\raw\ukraine_urls.txt
echo.
echo ETAPE 4: Lancer le projet
echo   
echo   Terminal 1 ^(Backend^):
echo     cd backend
echo     venv\Scripts\activate
echo     python main.py --auto
echo.
echo   Terminal 2 ^(Frontend^):
echo     cd frontend
echo     npm run dev
echo.
echo   Ouvrez: http://localhost:3000
echo.
echo ================================================================================
echo RACCOURCIS UTILES
echo ================================================================================
echo.
echo Double-cliquez sur ces fichiers pour lancer rapidement:
echo   - run_backend.bat  : Lance le backend
echo   - run_frontend.bat : Lance le frontend
echo.
echo ================================================================================
) > GUIDE_WINDOWS.txt

:: Raccourcis
(
echo @echo off
echo cd backend
echo call venv\Scripts\activate.bat
echo python main.py --auto
echo pause
) > run_backend.bat

(
echo @echo off
echo cd frontend
echo npm run dev
echo pause
) > run_frontend.bat

echo ✅ Guides Windows crees
echo.

echo ================================================================================
echo ✅ INSTALLATION TERMINEE !
echo ================================================================================
echo.
echo 📁 Projet cree: %PROJECT_NAME%\
echo.
echo 📖 Consultez:
echo    - README.md           : Documentation generale
echo    - GUIDE_WINDOWS.txt   : Guide specifique Windows
echo.
echo 🚀 Prochaines etapes:
echo.
echo 1. Telechargez les scripts Python depuis les artifacts
echo    et placez-les dans backend\scripts\
echo.
echo 2. Telechargez les fichiers frontend depuis les artifacts
echo    et placez-les dans frontend\app\
echo.
echo 3. Installez les dependances ^(voir GUIDE_WINDOWS.txt^)
echo.
echo 4. Ajoutez vos URLs dans:
echo    - backend\data\raw\gaza_urls.txt
echo    - backend\data\raw\ukraine_urls.txt
echo.
echo 5. Lancez:
echo    - run_backend.bat  ^(Terminal 1^)
echo    - run_frontend.bat ^(Terminal 2^)
echo.
echo ================================================================================
echo.
pause