# 📚 AI Technical Learning Assistant

A RAG-based AI learning assistant that helps users learn from technical articles and documentation.

The user provides a technical article URL, asks questions about the article, and can generate an MCQ quiz to test their understanding.

## 🚀 Features

- Load technical articles from URLs
- Split article content into smaller chunks
- Generate embeddings for the article
- Store embeddings in FAISS
- Ask questions about the article using RAG
- Retrieve relevant article content for each question
- Generate context-aware answers using Gemini
- Generate MCQ quizzes from the article
- Instant correct/incorrect feedback
- Explanation for every quiz answer
- Final quiz score

## 🧠 How It Works

### Document Processing

```text
Article URL
    ↓
WebBaseLoader
    ↓
Extract Document
    ↓
Text Chunking
    ↓
Gemini Embeddings
    ↓
FAISS Vector Store
