import streamlit as st

# =========================
# Title & intro
# =========================
st.title("Project Workflow")
st.caption("Follow these steps to conduct the NLP media bias analysis")

st.markdown("""
This workflow describes the complete research pipeline implemented in the project.
Each step corresponds to a concrete action performed in the backend.
""")

st.markdown("---")

# =========================
# Step 1 — Corpus
# =========================
with st.expander("1️⃣ Corpus Collection", expanded=True):
    st.markdown("""
    **Objective:** Build a reliable and comparable dataset.

    - Manual collection of article URLs
    - Focus on Western media coverage (AP News, BBC, CNN)
    - Separation of Gaza and Ukraine corpora
    """)
    if st.button("➡️ Go to Corpus Page"):
        st.switch_page("pages/1_Corpus.py")

# =========================
# Step 2 — Extraction
# =========================
with st.expander("2️⃣ Content Extraction"):
    st.markdown("""
    **Objective:** Retrieve raw textual content from collected URLs.

    - Python scripts extract article titles and body text
    - Metadata (URL, source, corpus label) is preserved
    - Output stored in raw JSON files
    """)
    st.code("backend/scripts/build_corpus.py")

# =========================
# Step 3 — Preprocessing
# =========================
with st.expander("3️⃣ Text Preprocessing"):
    st.markdown("""
    **Objective:** Clean and normalize text for NLP analysis.

    - Removal of non-linguistic elements (photo captions, copyright)
    - Lowercasing and whitespace normalization
    - Output: clean and analysis-ready corpora
    """)
    st.code("backend/scripts/clean_corpus.py")

# =========================
# Step 4 — NLP Analysis
# =========================
with st.expander("4️⃣ NLP Analysis"):
    st.markdown("""
    **Objective:** Detect linguistic asymmetries and framing differences.

    The following analyses are performed:
    - Lexical frequency analysis
    - Distinctive word analysis (log-ratio)
    - Semantic analysis (Word2Vec)
    - Sentiment analysis (VADER)
    """)
    st.code("""
scripts/lexical_analysis.py
scripts/distinctive_analysis.py
scripts/semantic_analysis.py
scripts/sentiment_analysis.py
""")

# =========================
# Step 5 — Visualization
# =========================
with st.expander("5️⃣ Visualization & Interpretation"):
    st.markdown("""
    **Objective:** Make results interpretable and communicable.

    - Generation of comparative plots
    - Export of figures for reports and frontend display
    - Interactive exploration via dashboard
    """)
    st.code("scripts/visualize_distinctive.py")
    if st.button("📊 View Analysis Results"):
        st.switch_page("pages/4_distinctive.py")

# =========================
# Notes
# =========================
st.markdown("---")
st.markdown("### ⚠️ Important Notes")

st.markdown("""
- Scripts are executed sequentially to preserve data integrity  
- Clean datasets are reused across all analyses  
- All steps are reproducible and modular  
- The pipeline can be extended to other conflicts or languages  
""")

st.info("""
This workflow ensures methodological rigor and allows the detection of media bias
at lexical, semantic, and emotional levels.
""")
