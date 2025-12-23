from app.services.corpus_builder import build_corpus

build_corpus(
    url_file="data/raw/gaza_urls.txt",
    output_file="data/raw/gaza_raw.json",
    label="gaza"
)

build_corpus(
    url_file="data/raw/ukraine_urls.txt",
    output_file="data/raw/ukraine_raw.json",
    label="ukraine"
)

