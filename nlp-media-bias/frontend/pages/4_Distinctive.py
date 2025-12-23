import streamlit as st
import json
from pathlib import Path

# =========================
# Load distinctive results (DYNAMIC)
# =========================
DISTINCTIVE_PATH = Path("../backend/data/results/distinctive_words.json")

with open(DISTINCTIVE_PATH, "r", encoding="utf-8") as f:
    distinctive = json.load(f)

# =========================
# Title & context
# =========================
st.title("Distinctive Vocabulary Analysis")
st.caption("Identifying words that distinguish Gaza and Ukraine coverage")

st.markdown("""
This section identifies **distinctive words** that are statistically more
associated with one corpus than the other.
Unlike raw frequency, this method highlights **relative specificity**.
""")

st.markdown("---")

# =========================
# Method explanation
# =========================
with st.expander("📐 How does distinctive analysis work?", expanded=True):
    st.markdown("""
    Distinctive words are computed using a **log-ratio** metric:

    - **Positive values** → words more characteristic of **Ukraine**
    - **Negative values** → words more characteristic of **Gaza**

    This approach controls for corpus size and highlights **systematic lexical framing**.
    """)

# =========================
# Ukraine distinctive
# =========================
st.markdown("## 🇺🇦 Distinctive Words — Ukraine")

st.markdown("""
Words associated with the Ukraine corpus emphasize:
- state actors and leadership
- military organization
- geopolitical framing
""")

st.image(
    "../backend/data/figures/ukraine_distinctive.png",
    caption="Top distinctive words in the Ukraine corpus",
    use_container_width=True
)

st.markdown("### 🔹 Top Distinctive Terms (Dynamic)")

for item in distinctive["ukraine"][:15]:
    st.markdown(f"- **{item['term']}** ({item['score']:.2f})")

# =========================
# Gaza distinctive
# =========================
st.markdown("## 🇵🇸 Distinctive Words — Gaza")

st.markdown("""
Words associated with the Gaza corpus emphasize:
- civilian population
- humanitarian conditions
- localized impact of violence
""")

st.image(
    "../backend/data/figures/gaza_distinctive.png",
    caption="Top distinctive words in the Gaza corpus",
    use_container_width=True
)

st.markdown("### 🔹 Top Distinctive Terms (Dynamic)")

for item in distinctive["gaza"][:15]:
    st.markdown(f"- **{item['term']}** ({item['score']:.2f})")

# =========================
# Interpretation
# =========================
st.markdown("---")
st.markdown("## Interpretation")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Ukraine coverage** tends to use:
    - institutional and geopolitical vocabulary
    - military hierarchy and strategy terms
    - leadership-focused framing
    """)

with col2:
    st.markdown("""
    **Gaza coverage** more frequently uses:
    - humanitarian and civilian-related terms
    - emotionally loaded descriptors
    - localized and individualized narratives
    """)

st.info("""
Distinctive vocabulary analysis provides quantitative evidence of framing differences,
which are further explored through semantic and sentiment analyses.
""")
