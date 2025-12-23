import streamlit as st
import json
from pathlib import Path
from collections import Counter

import re

STOP_TERMS = {
    "said", "file", "page", "refresh", "logged",
    "friday", "saturday", "sunday",
    "oct", "october", "november", "december","."
}

def is_valid_tfidf_term(term: str) -> bool:
    # remove pure numbers (years)
    if term.isdigit():
        return False

    # remove terms containing only digits
    if re.fullmatch(r"\d+", term):
        return False

    # remove generic journalistic noise
    if term.lower() in STOP_TERMS:
        return False

    return True


# =========================
# Load corpora
# =========================
with open("../backend/data/processed/ukraine_clean.json", encoding="utf-8") as f:
    ukraine = json.load(f)

with open("../backend/data/processed/gaza_clean.json", encoding="utf-8") as f:
    gaza = json.load(f)

def top_words(corpus, n=20):
    text = " ".join(a["text"] for a in corpus).lower()
    words = [w for w in text.split() if len(w) > 3]
    return Counter(words).most_common(n)

# =========================
# Title & context
# =========================
st.title("Lexical Analysis")
st.caption("Descriptive comparison of vocabulary usage across corpora")

st.markdown("""
Lexical analysis examines **word frequency** to identify dominant themes
and narrative focus in media coverage.  
At this stage, the analysis is **descriptive** and does not yet imply bias.
""")

st.markdown("---")

# =========================
# Explanation
# =========================
with st.expander("🔍 What does lexical frequency show?", expanded=True):
    st.markdown("""
    Word frequency analysis highlights:
    - recurrent topics
    - dominant actors
    - thematic focus of reporting
    
    This step provides context before more advanced comparative analyses.
    """)

# =========================
# Top words tables
# =========================
st.markdown("## Most Frequent Words")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🇺🇦 Ukraine")
    for word, count in top_words(ukraine):
        st.write(f"**{word}** — {count}")

with col2:
    st.subheader("🇵🇸 Gaza")
    for word, count in top_words(gaza):
        st.write(f"**{word}** — {count}")

st.markdown("## TF-IDF Keywords (Unigrams & Bigrams)")

st.markdown("""
This analysis highlights **lexical markers** that are characteristic of each corpus,
by reducing the impact of generic or high-frequency terms.
""")

# =========================
# Load TF-IDF results
# =========================
TFIDF_PATH = Path("../backend/data/results/tfidf_ngrams.json")

with open(TFIDF_PATH, "r", encoding="utf-8") as f:
    tfidf_data = json.load(f)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🇺🇦 Ukraine Coverage")
    for item in tfidf_data["ukraine"]:
        term = item["term"]
        if is_valid_tfidf_term(term):
            st.markdown(f"- **{term}**")


with col2:
    st.markdown("### 🇵🇸 Gaza Coverage")
    for item in tfidf_data["gaza"]:
        term = item["term"]
        if is_valid_tfidf_term(term):
            st.markdown(f"- **{term}**")



# =========================
# Interpretation
# =========================
st.markdown("---")
st.markdown("## Key Observations")

colA, colB = st.columns(2)

with colA:
    st.markdown("""
    **Ukraine coverage** is dominated by:
    - state and leadership terminology
    - military operations
    - territorial references
    """)

with colB:
    st.markdown("""
    **Gaza coverage** emphasizes:
    - armed groups
    - civilian population
    - humanitarian consequences
    """)

st.info("""
Lexical frequency alone does not establish bias.  
Subsequent analyses focus on **distinctiveness**, **semantic context**,
and **sentiment** to refine interpretation.
""")




