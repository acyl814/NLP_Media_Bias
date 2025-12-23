import streamlit as st
import json

# =========================
# Page configuration
# =========================
st.title("Corpus Overview")

st.markdown("""
This page presents the datasets used in the study.  
It provides transparency regarding data sources, corpus size, and examples
of the articles analyzed.
""")

# =========================
# Data sources
# =========================
st.markdown("## Data Sources")

st.markdown("""
Articles were collected from publicly accessible Western media outlets:

- **AP News**
- **BBC News**
- **CNN**

Only **English-language** articles were included to ensure consistency across
all NLP analyses.
""")

# =========================
# Load corpus data
# =========================
UKRAINE_PATH = "../backend/data/processed/ukraine_clean.json"
GAZA_PATH = "../backend/data/processed/gaza_clean.json"

with open(UKRAINE_PATH, "r", encoding="utf-8") as f:
    ukraine_data = json.load(f)

with open(GAZA_PATH, "r", encoding="utf-8") as f:
    gaza_data = json.load(f)

# =========================
# Corpus statistics
# =========================
st.markdown("## Corpus Statistics")

c1, c2 = st.columns(2)

with c1:
    st.metric("🇺🇦 Ukraine Articles", len(ukraine_data))
    st.metric("Language", "English")

with c2:
    st.metric("🇵🇸 Gaza Articles", len(gaza_data))
    st.metric("Time Period", "Oct 2023 – Present")

# =========================
# Article preview settings
# =========================
st.markdown("## Sample Articles Preview")
st.caption("Click on a title to expand and read a short excerpt")

n_preview = st.selectbox(
    "Number of articles to preview per corpus",
    [1, 2, 3, 5],
    index=2
)

# =========================
# Article previews
# =========================
col_u, col_g = st.columns(2)

# -------- Ukraine --------
with col_u:
    st.subheader("🇺🇦 Ukraine")

    for article in ukraine_data[:n_preview]:
        with st.expander(article["title"]):
            st.markdown("**Excerpt:**")
            st.write(article["text"][:600] + "...")
            st.markdown("**Metadata:**")
            st.write(f"- Corpus: {article['corpus']}")
            st.write(f"- Length: {len(article['text'].split())} words")
            st.markdown(
                f"[Read original article]({article['url']})",
                unsafe_allow_html=True
            )

# -------- Gaza --------
with col_g:
    st.subheader("🇵🇸 Gaza")

    for article in gaza_data[:n_preview]:
        with st.expander(article["title"]):
            st.markdown("**Excerpt:**")
            st.write(article["text"][:600] + "...")
            st.markdown("**Metadata:**")
            st.write(f"- Corpus: {article['corpus']}")
            st.write(f"- Length: {len(article['text'].split())} words")
            st.markdown(
                f"[Read original article]({article['url']})",
                unsafe_allow_html=True
            )

# =========================
# Limitations
# =========================
st.markdown("## Limitations")

st.markdown("""
- Corpus size is limited compared to large-scale media datasets.
- Only English-language articles were analyzed.
- Some media outlets restrict automated access to content.
- The corpus reflects mainstream Western media perspectives.
""")
