from app.services.lexical import (
    load_texts,
    tokenize_corpus,
    word_frequencies,
    ngram_frequencies
)

def analyze_corpus(name: str, file_path: str):
    texts = load_texts(file_path)
    tokens = tokenize_corpus(texts)

    print(f"\n===== {name.upper()} CORPUS =====")

    print("\nTop 30 mots les plus fréquents :")
    for word, freq in word_frequencies(tokens):
        print(f"{word:15} {freq}")

    print("\nTop 20 bigrammes :")
    for bigram, freq in ngram_frequencies(tokens, n=2):
        print(f"{bigram} -> {freq}")

    print("\nTop 20 trigrammes :")
    for trigram, freq in ngram_frequencies(tokens, n=3):
        print(f"{trigram} -> {freq}")


if __name__ == "__main__":
    analyze_corpus(
        "Ukraine",
        "data/processed/ukraine_clean.json"
    )

    analyze_corpus(
        "Gaza",
        "data/processed/gaza_clean.json"
    )
