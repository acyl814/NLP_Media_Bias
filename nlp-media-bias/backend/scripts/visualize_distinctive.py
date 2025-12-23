import matplotlib
matplotlib.use("Agg")  # backend non interactif (IMPORTANT)

import matplotlib.pyplot as plt
from app.services.distinctive import compute_frequencies, log_ratio
import os

OUTPUT_DIR = "data/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_distinctive_words(words, scores, title, filename):
    plt.figure(figsize=(10, 6))
    plt.barh(words, scores)
    plt.axvline(0, linestyle="--")
    plt.title(title)
    plt.xlabel("Log-ratio score")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))
    plt.close()

def run():
    ukraine_freq = compute_frequencies("data/processed/ukraine_clean.json")
    gaza_freq = compute_frequencies("data/processed/gaza_clean.json")

    top_ukraine, top_gaza = log_ratio(ukraine_freq, gaza_freq, top_n=15)

    plot_distinctive_words(
        [w for w, _ in top_ukraine],
        [s for _, s in top_ukraine],
        "Distinctive words — Ukraine corpus",
        "ukraine_distinctive.png"
    )

    plot_distinctive_words(
        [w for w, _ in top_gaza],
        [s for _, s in top_gaza],
        "Distinctive words — Gaza corpus",
        "gaza_distinctive.png"
    )

if __name__ == "__main__":
    run()
