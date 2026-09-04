import os

import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    create_vector_store,
    ask_question,
    generate_quiz
)


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

        # Reset previous quiz
        st.session_state.quiz = None
        st.session_state.quiz_index = 0
        st.session_state.quiz_score = 0
        st.session_state.quiz_submitted = False
        st.session_state.selected_answer = None

        st.success("✅ Article is ready to learn from!")

    except Exception as e:

        st.error(
            f"Error processing article: {e}"
        )


# =========================================================
# ASK QUESTIONS
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


# =========================================================
# QUIZ SECTION
# =========================================================

if "vector_store" in st.session_state:

    st.divider()

    st.subheader("📝 Test Your Knowledge")

    # -----------------------------------------------------
    # Generate Quiz
    # -----------------------------------------------------

    if st.session_state.get("quiz") is None:

        if st.button("🎯 Generate Quiz"):

            try:

                with st.spinner("Generating quiz from the article..."):

                    quiz = generate_quiz(
                        st.session_state.vector_store,
                        number_of_questions=5
                    )

                st.session_state.quiz = quiz
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_submitted = False
                st.session_state.selected_answer = None

                st.rerun()

            except Exception as e:

                st.error(
                    f"Error generating quiz: {e}"
                )


    # -----------------------------------------------------
    # Display Quiz
    # -----------------------------------------------------

    else:

        quiz = st.session_state.quiz
        current_index = st.session_state.quiz_index

        # Quiz completed
        if current_index >= len(quiz):

            score = st.session_state.quiz_score
            total = len(quiz)

            st.success(
                f"🎉 Quiz Complete! Your score is {score}/{total}"
            )

            percentage = int((score / total) * 100)

            st.write(
                f"### Score: {percentage}%"
            )

            if st.button("🔄 Take Quiz Again"):

                st.session_state.quiz = None
                st.session_state.quiz_index = 0
                st.session_state.quiz_score = 0
                st.session_state.quiz_submitted = False
                st.session_state.selected_answer = None

                st.rerun()

        else:

            current_question = quiz[current_index]

            st.write(
                f"### Question {current_index + 1} of {len(quiz)}"
            )

            st.write(
                f"**{current_question['question']}**"
            )

            # -------------------------------------------------
            # Answer selection
            # -------------------------------------------------

            selected_answer = st.radio(
                "Choose your answer:",
                current_question["options"],
                key=f"question_{current_index}"
            )

            # -------------------------------------------------
            # Submit Answer
            # -------------------------------------------------

            if not st.session_state.quiz_submitted:

                if st.button("Submit Answer"):

                    st.session_state.selected_answer = selected_answer
                    st.session_state.quiz_submitted = True

                    if (
                        selected_answer
                        == current_question["correct_answer"]
                    ):
                        st.session_state.quiz_score += 1

                    st.rerun()

            # -------------------------------------------------
            # Show Result
            # -------------------------------------------------

            if st.session_state.quiz_submitted:

                if (
                    st.session_state.selected_answer
                    == current_question["correct_answer"]
                ):

                    st.success("🟢 Correct!")

                else:

                    st.error("🔴 Incorrect!")

                    st.write(
                        f"**Correct answer:** "
                        f"{current_question['correct_answer']}"
                    )

                st.info(
                    f"**Explanation:** "
                    f"{current_question['explanation']}"
                )

                # -------------------------------------------------
                # Next Question
                # -------------------------------------------------

                if st.button("Next Question →"):

                    st.session_state.quiz_index += 1
                    st.session_state.quiz_submitted = False
                    st.session_state.selected_answer = None

                    st.rerun()