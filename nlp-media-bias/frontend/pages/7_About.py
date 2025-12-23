import streamlit as st

# =========================
# Title & intro
# =========================
st.title("About This Project")
st.caption("A comprehensive NLP research platform for analyzing media bias")

st.markdown("""
This project applies **Natural Language Processing (NLP)** techniques
to explore and compare how Western media outlets report on major conflicts.
The focus is on identifying **systematic differences in framing, vocabulary,
and emotional tone**.
""")

st.markdown("---")

# =========================
# Project objective
# =========================
st.markdown("## 🎯 Project Objective")

st.markdown("""
The primary objective of this project is to **detect and analyze potential media bias**
in Western news coverage through a comparative NLP approach.

The study focuses on two conflicts:
- **Gaza** (coverage since October 2023)
- **Ukraine** (coverage from the same media outlets)

Two complementary dimensions of bias are explored:
""")

st.markdown("""
- **Internal Bias Analysis**  
  How different actors are described *within the same conflict*
  (e.g., civilians vs armed groups).

- **Systemic Bias Analysis**  
  How coverage of Gaza differs from coverage of Ukraine in terms of
  narrative framing, vocabulary, and emotional tone.
""")

# =========================
# Methodology
# =========================
st.markdown("---")
st.markdown("## 🧪 Methodology")

st.markdown("""
The project follows a structured and reproducible NLP pipeline:
""")

st.markdown("""
**1. Data Collection**  
- Gaza corpus: ~50–100 English-language articles  
- Ukraine corpus: ~30–50 English-language articles  
- Sources: AP News, BBC, CNN  

**2. Lexical Analysis**  
- Identification of recurring terms  
- Comparison of vocabulary usage patterns  

**3. Distinctive Vocabulary Analysis**  
- Log-ratio based comparison  
- Detection of corpus-specific lexical framing  

**4. Semantic Analysis**  
- Word2Vec models trained separately on each corpus  
- Comparison of contextual meaning around key concepts  

**5. Sentiment Analysis**  
- Lexicon-based sentiment analysis (VADER)  
- Comparison of emotional tone and polarity  
""")

# =========================
# Expected findings / hypotheses
# =========================
st.markdown("---")
st.markdown("## 🔍 Analytical Hypotheses")

with st.container():
    st.markdown("### Internal Bias Hypothesis")
    st.markdown("""
    Media coverage may describe different actors within the same conflict
    using **asymmetric vocabulary**.

    For example:
    - Use of more **dehumanizing or technical terms** for some groups  
    - Use of more **empathetic or humanizing language** for others  

    This asymmetry can influence perceived responsibility and moral evaluation.
    """)

with st.container():
    st.markdown("### Systemic Bias Hypothesis")
    st.markdown("""
    Coverage of Ukraine may emphasize:
    - Heroic resistance  
    - Clear attribution of responsibility  
    - Moral clarity  

    In contrast, Gaza coverage may:
    - Use more neutral or abstract language  
    - Avoid explicit attribution of agency  
    - Present events as complex or ambiguous  

    These patterns may result in **false equivalence** or emotional distancing.
    """)

# =========================
# Contributions
# =========================
st.markdown("---")
st.markdown("## 📌 Contributions")

st.markdown("""
This project contributes:
- A **reproducible NLP pipeline** for media analysis  
- A **multi-level analytical framework** (lexical, semantic, sentiment)  
- A **transparent and interpretable dashboard** for exploratory analysis  
""")

# =========================
# Limitations
# =========================
st.markdown("---")
st.markdown("## ⚠️ Limitations")

st.markdown("""
- Corpus size is limited compared to large-scale datasets  
- Only English-language articles are analyzed  
- Automated sentiment analysis simplifies complex narratives  
- NLP methods do not infer author intent or editorial policy  
""")

# =========================
# Future work
# =========================
st.markdown("---")
st.markdown("## 🔮 Future Work")

st.markdown("""
Possible extensions of this project include:
- Expanding to additional conflicts or regions  
- Incorporating multilingual analysis  
- Using frame detection or topic modeling  
- Adding human annotation for validation  
""")

st.info("""
This project is intended as an exploratory research tool,
demonstrating how NLP can support critical media analysis
without replacing human interpretation.
""")
