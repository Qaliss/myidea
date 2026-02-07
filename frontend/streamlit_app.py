import streamlit as st
import requests

st.title("Idea Analyzer")

idea = st.text_input("Enter your idea:")
top_k = st.number_input("Top K papers", min_value=1, max_value=20, value=5)

if st.button("Analyze"):
    if not idea.strip():
        st.warning("Please enter an idea.")
    else:
        with st.spinner("Analyzing..."):
            response = requests.post(
                "http://127.0.0.1:8000/analyze_idea",
                json={"idea": idea, "top_k": top_k},
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
                )
