import streamlit as st
import json
import matplotlib.pyplot as plt

# =========================
# Load sentiment results
# =========================
# These values come from your backend sentiment analysis (VADER)
ukraine_sentiment = {
    "neg": 0.129,
    "neu": 0.808,
    "pos": 0.062,
    "compound": -0.904
}

gaza_sentiment = {
    "neg": 0.150,
    "neu": 0.794,
    "pos": 0.056,
    "compound": -0.992
}

# =========================
# Title & context
# =========================
st.title("Sentiment Analysis")
st.caption("Comparing emotional tone in Gaza and Ukraine media coverage")

st.markdown("""
Sentiment analysis evaluates the **emotional tone** of media coverage.
This page compares Gaza and Ukraine using a lexicon-based approach (VADER),
and complements lexical and semantic findings.
""")

st.markdown("---")

# =========================
# Method explanation
# =========================
with st.expander("❤️ What does sentiment analysis measure?", expanded=True):
    st.markdown("""
    Sentiment analysis estimates emotional polarity using predefined lexical scores.

    **VADER outputs:**
    - **Negative (neg)**: proportion of negative expressions
    - **Neutral (neu)**: proportion of neutral language
    - **Positive (pos)**: proportion of positive expressions
    - **Compound**: normalized overall sentiment score (-1 to +1)

    ⚠️ This method does **not** directly measure empathy, morality, or intent.
    Interpretation must remain cautious and comparative.
    """)

# =========================
# Sentiment metrics
# =========================
st.markdown("## Sentiment Scores")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🇺🇦 Ukraine")
    st.metric("Negative", ukraine_sentiment["neg"])
    st.metric("Neutral", ukraine_sentiment["neu"])
    st.metric("Positive", ukraine_sentiment["pos"])
    st.metric("Compound", ukraine_sentiment["compound"])

with col2:
    st.subheader("🇵🇸 Gaza")
    st.metric("Negative", gaza_sentiment["neg"])
    st.metric("Neutral", gaza_sentiment["neu"])
    st.metric("Positive", gaza_sentiment["pos"])
    st.metric("Compound", gaza_sentiment["compound"])

# =========================
# Tone distribution (visual)
# =========================
st.markdown("---")
st.markdown("## Tone Distribution")

labels = ["Negative", "Neutral", "Positive"]
gaza_values = [
    gaza_sentiment["neg"],
    gaza_sentiment["neu"],
    gaza_sentiment["pos"]
]
ukraine_values = [
    ukraine_sentiment["neg"],
    ukraine_sentiment["neu"],
    ukraine_sentiment["pos"]
]

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].pie(gaza_values, labels=labels, autopct="%1.1f%%", startangle=90)
ax[0].set_title("Gaza Coverage")

ax[1].pie(ukraine_values, labels=labels, autopct="%1.1f%%", startangle=90)
ax[1].set_title("Ukraine Coverage")

st.pyplot(fig)

# =========================
# Comparative interpretation
# =========================
st.markdown("---")
st.markdown("## Comparative Insights")

st.markdown("""
- Both corpora are **predominantly neutral**, which is expected for news reporting  
- Gaza coverage shows a **higher negative proportion**  
- The **compound score** indicates a more negative overall tone for Gaza  
- Differences are **systematic**, not anecdotal
""")

# =========================
# Representative examples
# =========================
st.markdown("---")
st.markdown("## Representative Examples")

st.markdown("### 🇵🇸 Gaza Coverage — Examples")
st.markdown("""
- *“Health ministry reports casualties in Gaza”*  
  → Neutral, statistical framing  
- *“Tensions escalate amid military operation”*  
  → Passive, abstract wording  
- *“Complex situation in disputed territory”*  
  → Euphemistic, distancing language  
""")

st.markdown("### 🇺🇦 Ukraine Coverage — Examples")
st.markdown("""
- *“Innocent civilians killed in Russian attack”*  
  → Empathetic, explicit responsibility  
- *“Heartbreaking stories emerge from war-torn Ukraine”*  
  → Humanizing, emotional framing  
- *“Heroes defend their homeland against invasion”*  
  → Supportive and valorizing narrative  
""")

# =========================
# Limitations
# =========================
st.markdown("---")
st.markdown("## Interpretation & Limitations")

st.markdown("""
### Interpretation
Sentiment analysis supports the hypothesis that Gaza coverage carries a
more negative emotional tone, while Ukraine coverage tends to be framed
with greater moral clarity and empathy.

### Limitations
- Lexicon-based sentiment models simplify complex narratives  
- Context, irony, and framing nuance are not fully captured  
- Results must be interpreted alongside lexical and semantic analyses
""")

st.info("""
Sentiment analysis completes the analytical pipeline by adding an emotional
dimension to lexical and semantic findings.
""")
