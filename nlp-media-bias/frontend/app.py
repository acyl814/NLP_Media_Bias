import streamlit as st


st.set_page_config(
    page_title="MediaBias NLP",
    layout="wide"
)

# =========================
# Global CSS (apply once)
# =========================
st.markdown("""
<style>

/* Global */
html, body, [class*="css"]  {
    font-family: "Inter", "Segoe UI", sans-serif;
}

/* Titles */
h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 1.6rem;
    font-weight: 600;
    margin-top: 2rem;
}

h3 {
    font-size: 1.2rem;
    font-weight: 600;
}

/* Captions */
.small-text {
    color: #6b7280;
    font-size: 0.9rem;
}

/* Cards */
.card {
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    margin-bottom: 1.5rem;
}

/* Highlight boxes */
.highlight {
    background-color: #f9fafb;
    padding: 1rem;
    border-left: 4px solid #2563eb;
    border-radius: 6px;
    margin-top: 1rem;
}

/* Info boxes */
.info-box {
    background-color: #f0f7ff;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dbeafe;
}

/* Buttons */
button[kind="primary"] {
    background-color: #2563eb;
    border-radius: 8px;
}

/* Metrics */
div[data-testid="metric-container"] {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 1rem;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# Header
st.markdown("### 📊 MediaBias NLP")

st.markdown(
    """
    <h1 style='text-align: center;'>
    Detecting Double Standards<br>in Media Coverage
    </h1>
    <p style='text-align: center; font-size:18px;'>
    A comprehensive NLP platform for analyzing bias in Western media coverage<br>
    of <b>Gaza</b> and <b>Ukraine</b> conflicts.
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    cta1, cta2 = st.columns(2)

    if cta1.button("🚀 Start Analysis"):
        st.switch_page("pages/1_Corpus.py")

    if cta2.button("📘 Learn More"):
        st.switch_page("pages/7_About.py")

st.markdown("<br><br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

c1.metric("📰 Gaza Articles", "50–100")
c2.metric("📰 Ukraine Articles", "30–50")
c3.metric("🔍 Analysis Types", "3")
c4.metric("⚙️ Processing", "Offline NLP")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## Research Workflow")
st.caption("From data collection to insights")

wf1, wf2, wf3, wf4 = st.columns(4)

wf1.markdown("### 1️⃣ Corpus\nCollect articles from major Western media.")
wf2.markdown("### 2️⃣ Preprocessing\nClean and normalize textual data.")
wf3.markdown("### 3️⃣ Analysis\nLexical, semantic and sentiment analysis.")
wf4.markdown("### 4️⃣ Visualization\nClear figures and comparisons.")

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("## Analysis Modules")
st.caption("Comprehensive tools for detecting media bias")

m1, m2, m3 = st.columns(3)

m1.markdown("""
### 🔤 Lexical Analysis
Identify recurring terms and vocabulary asymmetries across conflicts.
""")

m2.markdown("""
### 🧠 Semantic Analysis
Study contextual usage and reveal differentiated interpretive frameworks.
""")

m3.markdown("""
### ❤️ Sentiment Analysis
Analyze emotional framing and quantify empathy disparities.
""")

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align:center; padding:30px; border:1px solid #ddd; border-radius:10px;'>
        <h3>Ready to Start Your Analysis?</h3>
        <p>Build your corpus and uncover media bias patterns with a complete NLP pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

colA, colB, colC = st.columns([1,2,1])
with colB:
    if st.button("▶️ Get Started"):
        st.switch_page("pages/1_Corpus.py")
