import json
from app.services.cleaner import clean_text

def clean_corpus(input_file: str, output_file: str):
    with open(input_file, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    cleaned_corpus = []

    for article in corpus:
        cleaned_article = {
            "id": article["id"],
            "corpus": article["corpus"],
            "url": article["url"],
            "title": clean_text(article["title"]),
            "text": clean_text(article["text"])
        }
        cleaned_corpus.append(cleaned_article)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cleaned_corpus, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    clean_corpus(
        input_file="data/raw/ukraine_raw.json",
        output_file="data/processed/ukraine_clean.json"
    )

    clean_corpus(
        input_file="data/raw/gaza_raw.json",
        output_file="data/processed/gaza_clean.json"
    )
