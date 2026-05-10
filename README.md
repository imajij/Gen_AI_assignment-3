# NotebookLM-Style RAG Application

A production-ready Retrieval-Augmented Generation (RAG) system inspired by Google NotebookLM.

This application allows users to upload PDF or text documents and interact with them conversationally through semantic search, document chunking, vector embeddings, and grounded LLM-generated responses.

---

## Features

- Upload PDF, TXT, DOCX, Markdown, or HTML files
- Intelligent document parsing and text extraction
- Recursive chunking with overlap preservation
- Local embedding generation using SentenceTransformers
- ChromaDB vector storage for semantic retrieval
- OpenAI-compatible API architecture
- Supports configurable providers:
  - Google Gemini OpenAI-compatible endpoints
  - OpenRouter
  - TogetherAI
  - Ollama / LM Studio
- Grounded document Q&A
- Session-based isolated notebooks
- Streamlit-powered modern UI
- Deployable on Streamlit Community Cloud

---

## Tech Stack

| Layer | Technology |
|------|------------|
| Frontend | Streamlit |
| Backend | Python |
| Vector Database | ChromaDB |
| Embeddings | all-MiniLM-L6-v2 |
| LLM | OpenAI-Compatible API |
| Deployment | Streamlit Community Cloud |

---

## Project Structure

```txt
NotebookLM-RAG/
│
├── app.py
├── requirements.txt
├── README.md
├── .env.example
│
└── src/
    ├── config.py
    ├── document_loader.py
    ├── text_chunker.py
    ├── embedding_store.py
    ├── retriever.py
    ├── llm_handler.py
    ├── prompt_templates.py
    ├── validators.py
    ├── utils.py
    └── rag_pipeline.py
````

---

## How It Works

### Document Ingestion Pipeline

1. User uploads document
2. File is parsed into text
3. Text is chunked intelligently
4. Chunks are embedded
5. Embeddings are stored in ChromaDB

### Query Pipeline

1. User asks question
2. Query is embedded
3. Semantic search retrieves relevant chunks
4. Retrieved context is injected into prompt
5. LLM generates grounded answer

---

## Environment Variables

Create `.env`:

```env
BASE_URL=your_openai_compatible_endpoint
API_KEY=your_api_key
MODEL=gemma-4-26b-it
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_PATH=/tmp/vector_store
```

---

## Local Setup

```bash
git clone https://github.com/imajij/Gen_AI_assignment-3.git
cd Gen_AI_assignment-3

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
```

---

## Deployment

This project is optimized for Streamlit Community Cloud using temporary vector storage.

* Persistent long-term storage is NOT required
* Session data is automatically ephemeral
* ChromaDB uses `/tmp/vector_store`
* Old sessions are naturally removed on app restarts

---

## Security Features

* API keys stored securely in Streamlit secrets
* Session-isolated collections
* Prompt grounding
* Hallucination prevention
* Temporary storage design

---

## Academic Highlights

This project demonstrates:

* End-to-end RAG architecture
* Document retrieval systems
* Vector databases
* Semantic search
* Prompt engineering
* LLM grounding
* Real-world deployment

---

## Deployment Link

genaiassignment-3.streamlit.app

---

## Author

Ajij Uttam
