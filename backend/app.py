import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import create_vector_store, ask_question


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Technical Learning Assistant",
    page_icon="📚",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📚 AI Technical Learning Assistant")

st.write(
    "Paste a technical article or documentation URL "
    "and learn from it using RAG."
)


# =========================================================
# URL INPUT
# =========================================================

url = st.text_input(
    "Enter article/documentation URL:"
)


# =========================================================
# ANALYZE ARTICLE
# =========================================================

if st.button("Analyze Article"):

    if not url:
        st.warning("Please enter a URL.")
        st.stop()

    try:

        with st.spinner("Processing article..."):

            vector_store = create_vector_store(url)

        st.session_state.vector_store = vector_store
        st.session_state.url = url

        st.success("✅ Article is ready to learn from!")

    except Exception as e:

        st.error(
            f"Error processing article: {e}"
        )


# =========================================================
# QUESTION SECTION
# =========================================================

if "vector_store" in st.session_state:

    st.divider()

    st.subheader("💬 Ask Questions About the Article")

    question = st.text_input(
        "Ask a question:"
    )

    if st.button("Ask"):

        if not question:
            st.warning("Please enter a question.")
            st.stop()

        try:

            with st.spinner("Generating answer..."):

                answer = ask_question(
                    st.session_state.vector_store,
                    question
                )

            st.subheader("🤖 Answer")

            st.write(answer)

        except Exception as e:

            st.error(
                f"Error generating answer: {e}"
            )