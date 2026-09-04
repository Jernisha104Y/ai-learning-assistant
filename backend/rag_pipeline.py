import os
import json

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# =========================================================
# CREATE VECTOR STORE
# =========================================================

def create_vector_store(url):

    # 1. Load webpage
    loader = WebBaseLoader(url)

    documents = loader.load()

    # 2. Split document into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    # 3. Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # 4. Store embeddings in FAISS
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store


# =========================================================
# ASK QUESTION
# =========================================================

def ask_question(vector_store, question):

    # Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(
        question,
        k=4
    )

    # Combine retrieved content
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # Create prompt
    prompt = f"""
You are an AI technical learning assistant.

Answer the user's question using ONLY the
provided article context.

If the answer is not available in the context,
say that the information is not available
in the provided article.

ARTICLE CONTEXT:
{context}

USER QUESTION:
{question}

Give a clear and beginner-friendly answer.
"""

    # Ask Gemini
    response = llm.invoke(prompt)

    # Extract text
    if isinstance(response.content, str):

        answer = response.content

    else:

        answer = "\n".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    return answer


# =========================================================
# GENERATE QUIZ
# =========================================================

def generate_quiz(vector_store, number_of_questions=5):

    # Retrieve article content
    retrieved_docs = vector_store.similarity_search(
        "main concepts, important definitions, key points and important facts from the article",
        k=8
    )

    # Combine retrieved content
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # Create Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # Prompt Gemini
    prompt = f"""
You are an AI technical learning assistant.

Create exactly {number_of_questions} multiple-choice
questions based ONLY on the provided article context.

Each question must have:
- question
- four options
- correct_answer
- explanation

The correct_answer must be the exact option text.

Return ONLY valid JSON.

Use this exact format:

[
    {{
        "question": "Question text",
        "options": [
            "Option A",
            "Option B",
            "Option C",
            "Option D"
        ],
        "correct_answer": "Option B",
        "explanation": "Explanation of why this answer is correct."
    }}
]

Do not include markdown.
Do not include ```json.
Do not add any text outside the JSON.

ARTICLE CONTEXT:
{context}
"""

    # Ask Gemini
    response = llm.invoke(prompt)

    # Extract response text
    if isinstance(response.content, str):

        quiz_text = response.content

    else:

        quiz_text = "\n".join(
            block.get("text", "")
            for block in response.content
            if isinstance(block, dict)
            and block.get("type") == "text"
        )

    # Remove accidental markdown formatting
    quiz_text = quiz_text.strip()

    if quiz_text.startswith("```json"):
        quiz_text = quiz_text[7:]

    if quiz_text.startswith("```"):
        quiz_text = quiz_text[3:]

    if quiz_text.endswith("```"):
        quiz_text = quiz_text[:-3]

    quiz_text = quiz_text.strip()

    # Convert JSON text into Python object
    quiz = json.loads(quiz_text)

    return quiz