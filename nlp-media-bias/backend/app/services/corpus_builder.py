import json
from app.services.scraper import scrape_article
from app.services.url_loader import load_urls

def build_corpus(url_file: str, output_file: str, label: str):
    urls = load_urls(url_file)
    corpus = []

    for i, url in enumerate(urls):
        try:
            article = scrape_article(url)
            article["id"] = f"{label}_{i+1}"
            article["corpus"] = label
            corpus.append(article)
        except Exception as e:
            print(f"Erreur avec {url}: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)
