import json
from pathlib import Path
import streamlit as st

# =========================
# Load semantic results
# =========================
SEMANTIC_PATH = Path("../backend/data/results/semantic_similarities.json")

with open(SEMANTIC_PATH, "r", encoding="utf-8") as f:
    semantic = json.load(f)

st.markdown("## Key Concept Comparisons")

# =========================
# Helper function
# =========================
def render_semantic_block(title, left_label, left_items, right_label, right_items):
    st.markdown(f"### {title}")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🇺🇦 Ukraine")
        st.markdown(f"**{left_label} →**")
        for item in left_items:
            st.markdown(f"- {item['term']}")

    with col2:
        st.subheader("🇵🇸 Gaza")
        st.markdown(f"**{right_label} →**")
        for item in right_items:
            st.markdown(f"- {item['term']}")

# =========================
# Soldiers / Militants
# =========================
render_semantic_block(
    "🪖 Soldiers / Militants",
    "soldiers",
    semantic["ukraine"]["soldiers"],
    "militants",
    semantic["gaza"]["militants"]
)

# =========================
# Civilians / Children
# =========================
render_semantic_block(
    "👥 Civilians / Children",
    "civilians",
    semantic["ukraine"]["civilians"],
    "children",
    semantic["gaza"]["children"]
)

# =========================
# Attacks
# =========================
render_semantic_block(
    "💥 Attacks",
    "attack",
    semantic["ukraine"]["attack"],
    "attack",
    semantic["gaza"]["attack"]
)
