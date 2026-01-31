# ArXiv Intelligent RAG Search

A local Retrieval-Augmented Generation (RAG) application that allows users to semantically search thousands of ArXiv research papers and get AI-synthesized answers.

## 🏗️ Architecture

1.  **Data Processing:** Ingests raw ArXiv metadata JSON.
2.  **Embedding:** Uses `SentenceTransformers` (all-MiniLM-L6-v2) to convert abstracts into vectors.
3.  **Vector Database:** Stores embeddings locally using `ChromaDB` for semantic retrieval.
4.  **LLM Synthesis:** Uses Google's `Gemini Pro` API to generate answers based on retrieved context.
5.  **Frontend:** Built with `Streamlit`.

## ✅ Prerequisites

1.  Python 3.10+
2.  A Google AI Studio API Key (Free).

## 🚀 Setup Instructions

Since the raw data and database are too large for GitHub, you must build the index locally first.

**1. Clone the repository:**
```bash
git clone [YOUR_GITHUB_REPO_LINK_HERE]
cd arxiv-rag-search
