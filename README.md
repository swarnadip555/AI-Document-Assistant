# DocPilot AI — Intelligent AI Document Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-00A67E.svg)](https://www.langchain.com/)
[![FAISS](https://img.shields.io/badge/FAISS-VectorStore-orange.svg)](https://github.com/facebookresearch/faiss)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, glassmorphic **Retrieval-Augmented Generation (RAG)** web application for asking questions, synthesizing executive summaries, and generating multiple-choice quizzes from PDF documents.

Inspired by modern cybersecurity and cloud intelligence dashboards, **DocPilot AI** features a futuristic dark-mode glassmorphic interface, precise page-level citations, document metadata stats, and dual-mode AI inference (offline extractive RAG and Google Gemini cloud integration).

---

## 🌟 Key Features

1. **📄 PDF Processing & Extraction**:
   - Accepts text-based PDF documents.
   - Extracts page-level text, character counts, and document metadata using `pypdf`.
   - Handles corrupted or empty files gracefully.

2. **🧩 Semantic Chunking & Vector Search**:
   - Splits document pages into overlapping semantic chunks with `RecursiveCharacterTextSplitter`.
   - Generates normalized text embeddings via `sentence-transformers/all-MiniLM-L6-v2`.
   - Persists dense vectors locally using the `FAISS` vector database.

3. **💬 RAG Question Answering**:
   - Dense semantic retrieval with similarity scoring.
   - Generates grounded answers based strictly on retrieved document context.
   - Highlights exact source citations (`[Source: Page X]`).
   - Retains conversation history throughout the active session.

4. **📑 Document Summarization**:
   - Produces executive summaries, key takeaways, and section breakdowns.
   - Quotes source pages for each key insight.

5. **🎯 Interactive MCQ Quiz Generator**:
   - Generates 4-option multiple-choice questions from the uploaded PDF.
   - Interactive quiz interface with instant answer validation and explanations.
   - Live score tracking (e.g. 4/4 - 100%).

6. **💎 Minimal & Glassmorphic UI**:
   - Deep cyber slate background (`#080c14`).
   - Frosted glass cards (`backdrop-filter: blur(14px)`).
   - Glowing cyan accents, document processing metrics, and mini visual sparklines.

---

## 📂 Project Structure

```text
AI-Document-Assistant/
├── app.py                     # Main Streamlit web application & UI
├── requirements.txt           # Project dependencies
├── config.py                  # Global configurations & constants
├── README.md                  # Documentation
│
├── data/
│   └── uploaded_pdfs/         # Stored uploaded PDF documents
│
├── vectorstore/               # Local FAISS index persistence (index.faiss, index.pkl)
│
├── modules/
│   ├── __init__.py
│   ├── document_loader.py     # PDF parsing & metadata extraction
│   ├── text_splitter.py       # Semantic text chunking
│   ├── embeddings.py          # Sentence-Transformers embedding loader
│   ├── vector_database.py     # FAISS vector store creation & persistence
│   ├── retriever.py           # Similarity retrieval & citation formatting
│   ├── chatbot.py             # RAG pipeline & LLM generation
│   ├── summarizer.py          # Document summarization
│   └── mcq_generator.py       # 4-option MCQ generator & quiz engine
│
└── utils/
    ├── __init__.py
    └── helpers.py             # Custom glass CSS, file helpers & sample PDF generator
```

---

## 🚀 Quickstart & Installation

### 1. Activate your virtual environment
On Windows:
```powershell
.\venv\Scripts\activate
```

### 2. Install dependencies
```powershell
pip install -r requirements.txt
```

### 3. Launch the Streamlit application
```powershell
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 🧪 Testing with the Built-in Sample PDF

No PDF on hand? Simply click the **"🧪 Sample PDF"** button in the sidebar. This will instantly create and index a 4-page enterprise cloud report (*"Q3 Technical Report: Cloud Security and Compliance"*), allowing you to test:
- Page exploration & metadata stats
- Asking: *"What are the primary recommendations for optimizing cloud infrastructure security?"*
- Asking: *"What MFA protocols are enforced?"*
- Generating an Executive Summary
- Playing the interactive 4-question quiz

