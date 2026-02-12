# 🌐 RAG Online — Web Link Ingestion

A standalone RAG (Retrieval-Augmented Generation) system for **web content**.

Paste a URL (article, blog post, or web-hosted PDF) and the system will scrape/download the content, chunk it, embed it, and let you ask questions — all running locally.

---

## ✨ Features

- **Web PDF download** — automatically downloads and extracts PDFs from URLs
- **Web article scraping** — extracts readable text from blogs, news, Medium, etc.
- **Local file path support** — also accepts paths to local PDF files
- **Local LLM** — powered by [Ollama](https://ollama.com) (no API keys needed)
- **Local embeddings** — HuggingFace `all-MiniLM-L6-v2` runs on your machine
- **Chat interface** — ask questions and get grounded answers with source citations

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| LLM | Ollama (gemma3:4b) |
| Embeddings | HuggingFace sentence-transformers |
| Vector Store | ChromaDB |
| PDF Parsing | pypdf |
| Web Scraping | BeautifulSoup + lxml |
| HTTP Client | requests |
| Orchestration | LangChain |

## 📁 Project Structure

```
rag_online/
├── main.py              ← FastAPI app + scraping + RAG pipeline
├── templates/
│   └── index.html       ← Web UI (URL input + chat)
├── requirements.txt     ← Python dependencies
├── chroma_db_online/    ← (auto-created) Vector database storage
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
cd rag_online

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --port 8002
```

Open **http://localhost:8002** in your browser.

### Usage

1. **Paste a link** — Enter a URL (article, PDF link, or local path)
2. **Click Ingest** — The system detects the source type and extracts text
3. **Ask** — Type a question in the chat box and press Enter
4. **Review** — The answer appears with source citations

### Supported Source Types

| Type | Example | How It's Processed |
|---|---|---|
| **Web PDF** | `https://example.com/paper.pdf` | Downloaded → pypdf extraction |
| **Web Article** | `https://medium.com/@user/post` | Scraped with BeautifulSoup |
| **Local PDF** | `/path/to/file.pdf` | Read from disk → pypdf extraction |

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/ingest` | Ingest a link (JSON: `{"link": "..."}` or form data) |
| `POST` | `/api/query` | Ask a question (JSON: `{"question": "..."}`) |
| `GET` | `/api/status` | Get ingestion stats |

## ⚙️ Configuration

Edit the constants at the top of `main.py`:

| Parameter | Default | Description |
|---|---|---|
| `CHROMA_PERSIST_DIR` | `./chroma_db_online` | ChromaDB storage path |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `OLLAMA_MODEL_NAME` | `gemma3:4b` | Ollama LLM model |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `10` | Chunks retrieved per query |
