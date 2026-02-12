# 📄 RAG Local — PDF Upload

A standalone RAG (Retrieval-Augmented Generation) system for **local PDF files**.

Upload PDFs through the web UI, and the system will extract text, chunk it, embed it into a vector database, and let you ask questions about the content — all running locally.

---

## ✨ Features

- **Drag & drop PDF upload** — intuitive file upload with instant feedback
- **Local LLM** — powered by [Ollama](https://ollama.com) (no API keys needed)
- **Local embeddings** — HuggingFace `all-MiniLM-L6-v2` runs on your machine
- **Persistent storage** — ChromaDB stores embeddings to disk
- **Chat interface** — ask questions and get grounded answers with source citations

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| LLM | Ollama (gemma3:4b) |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | ChromaDB |
| PDF Parsing | pypdf |
| Orchestration | LangChain |

## 📁 Project Structure

```
rag_local/
├── main.py              ← FastAPI app + PDF extraction + RAG pipeline
├── templates/
│   └── index.html       ← Web UI (drag-drop upload + chat)
├── requirements.txt     ← Python dependencies
├── chroma_db_local/     ← (auto-created) Vector database storage
└── README.md
```

## 🚀 Getting Started

### Prerequisites

1. **Python 3.10+**
2. **Ollama** — Install from [ollama.com](https://ollama.com), then pull the model:
   ```bash
   ollama pull gemma3:4b
   ```

### Installation

```bash
# Navigate to the project
cd rag_local

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --port 8001
```

Open **http://localhost:8001** in your browser.

### Usage

1. **Upload** — Drag & drop a PDF file (or click to browse)
2. **Wait** — The system extracts text, chunks it, and embeds it
3. **Ask** — Type a question in the chat box and press Enter
4. **Review** — The answer appears with source citations

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/ingest` | Upload a PDF file |
| `POST` | `/api/query` | Ask a question (JSON: `{"question": "..."}`) |
| `GET` | `/api/status` | Get ingestion stats |

## ⚙️ Configuration

Edit the constants at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `./chroma_db_local` | ChromaDB storage path |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `OLLAMA_MODEL_NAME` | `gemma3:4b` | Ollama LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `10` | Chunks retrieved per query |
