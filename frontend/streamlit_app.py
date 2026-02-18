import streamlit as st
import requests
from sentence_transformers import SentenceTransformer

st.title("Idea Analyzer")

# Load model once (cached)
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

idea = st.text_input("Enter your idea:")
top_k = st.number_input("Top K papers", min_value=1, max_value=20, value=5)

if st.button("Analyze"):
    if not idea.strip():
        st.warning("Please enter an idea.")
    else:
        with st.spinner("Embedding idea..."):
            embedding = model.encode(
                idea,
                normalize_embeddings=True
            ).tolist()

        with st.spinner("Searching papers..."):
            response = requests.post(
                "https://myidea-8siz.onrender.com/analyze_idea",
                json={"embedding": embedding, "top_k": top_k},
                timeout=30
            )

        if response.status_code != 200:
            st.error(f"Backend error: {response.text}")
        else:
            papers = response.json()

            st.subheader("Relevant Papers")
            for p in papers:
                st.markdown(f"### {p['title']}")
                st.write(p["abstract"])
                st.caption(
                    f"Score: {p['score']:.3f} | "
                    f"Published: {p['published_date']}"
                    f"Link: {p['arxiv_id']}"
                )
