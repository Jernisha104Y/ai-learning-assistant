import os

from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS


# Load environment variables
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

    # 5. Retrieve relevant chunks
    retrieved_docs = vector_store.similarity_search(
        question,
        k=4
    )

    # 6. Combine retrieved chunks
    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )

    # 7. Create Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash"
    )

    # 8. Create prompt
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

    # 9. Ask Gemini
    response = llm.invoke(prompt)

    # 10. Extract answer
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