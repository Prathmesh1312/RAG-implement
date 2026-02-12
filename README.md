# 🧠 RAG Workshop — Demo Projects

A collection of **3 standalone RAG (Retrieval-Augmented Generation) projects** designed for workshop demonstrations, learning, and experimentation.

Each project is self-contained with its own backend, web UI, and dependencies.

---

## 📂 Projects

| # | Project | Port | Description |
|---|---|---|---|
| 1 | [📄 RAG Local](./rag_local/) | `8001` | Upload local PDF files and ask questions |
| 2 | [🌐 RAG Online](./rag_online/) | `8002` | Paste web links (articles, PDFs) and ask questions |
| 3 | [✂️ Chunking Demo](./rag_chunking/) | `8003` | Visualize and compare 7 chunking strategies |

---

### 📄 RAG Local — `rag_local/`

Upload PDF files through a drag-and-drop interface. The system extracts text, chunks it, embeds it into ChromaDB, and answers your questions using a local LLM via Ollama.

**Key:** PDF upload → text extraction → chunking → embedding → Q&A

```bash
cd rag_local && uvicorn main:app --reload --port 8001
```

---

### 🌐 RAG Online — `rag_online/`

Paste a URL — web article, blog post, or hosted PDF — and the system scrapes/downloads the content, processes it through the RAG pipeline, and lets you chat with it.

**Key:** URL paste → auto-detect type → scrape/download → chunking → embedding → Q&A

```bash
cd rag_online && uvicorn main:app --reload --port 8002
```

---

### ✂️ Chunking Demo — `rag_chunking/`

Upload a PDF and **visually compare** how 7 different chunking strategies split the same document. No LLM needed — this is a pure visualization tool.

**Strategies:** Fixed-size · Variable-size · Content-based · Logical · Dynamic · File-based · Task-based

```bash
cd rag_chunking && uvicorn main:app --reload --port 8003
```

---

## 🚀 Quick Start

### Prerequisites

| Requirement | For | Installation |
|---|---|---|
| **Python 3.10+** | All projects | [python.org](https://python.org) |
| **Ollama** | RAG Local & Online | [ollama.com](https://ollama.com) |
| **gemma3:4b model** | RAG Local & Online | `ollama pull gemma3:4b` |

> **Note:** The Chunking Demo does not require Ollama or any LLM.

### Run Any Project

```bash
# 1. Navigate to a project
cd rag_local    # or rag_online, rag_chunking

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start the server
uvicorn main:app --reload --port 8001   # 8002 for online, 8003 for chunking
```

### Run All Projects Simultaneously

```bash
# Terminal 1
cd rag_local && uvicorn main:app --reload --port 8001

# Terminal 2
cd rag_online && uvicorn main:app --reload --port 8002

# Terminal 3
cd rag_chunking && uvicorn main:app --reload --port 8003
```

## 🏗️ Tech Stack

| Component | Technology | Used In |
|---|---|---|
| Web Framework | FastAPI | All |
| LLM | Ollama (gemma3:4b) | Local, Online |
| Embeddings | HuggingFace sentence-transformers | Local, Online |
| Vector Store | ChromaDB | Local, Online |
| PDF Parsing | pypdf | All |
| Web Scraping | BeautifulSoup + lxml | Online |
| UI | Vanilla HTML/CSS/JS | All |
| Orchestration | LangChain | Local, Online |

## 📁 Folder Structure

```
rag1.5/
├── README.md               ← You are here
│
├── rag_local/              ← Project 1: Local PDF Upload
│   ├── main.py
│   ├── templates/index.html
│   ├── requirements.txt
│   └── README.md
│
├── rag_online/             ← Project 2: Web Link Ingestion
│   ├── main.py
│   ├── templates/index.html
│   ├── requirements.txt
│   └── README.md
│
└── rag_chunking/           ← Project 3: Chunking Visualization
    ├── main.py
    ├── templates/index.html
    ├── requirements.txt
    └── README.md
```
