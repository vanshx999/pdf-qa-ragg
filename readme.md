# PDF Q&A RAG System

Live demo: [YOUR_URL_HERE]

A production-ready RAG pipeline that lets you upload PDFs and ask questions via a web interface.

## What It Does

- Upload PDFs via drag-and-drop
- Ask natural language questions about the content
- Get answers with source citations (which chunks were used)
- Hybrid retrieval: BM25 keyword search + dense vector similarity

## Tech Stack

- **Backend:** FastAPI, ChromaDB, sentence-transformers, Groq API (Llama 3)
- **Frontend:** HTML + vanilla JavaScript (fetch API)
- **Deployment:** Docker, AWS EC2

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload a PDF file |
| `/ask` | POST | Ask a question about uploaded PDFs |
| `/health` | GET | Health check |

## Quick Start (Local)

```bash
git clone https://github.com/vanshx999/pdf-qa-rag.git
cd pdf-qa-rag
docker build -t pdf-qa-rag .
docker run -p 8000:8000 pdf-qa-rag