# ✂️ RAG Chunking Demo — Visualize 7 Strategies

An interactive tool to **visualize and compare** how 7 different chunking strategies split the same document. Upload a PDF and see the results side by side.

---

## ✨ Features

- **7 chunking strategies** — each with distinct logic and use cases
- **Visual chunk cards** — color-coded with character/word counts and size bars
- **Configurable parameters** — adjust chunk size, overlap, workers via UI sliders
- **Compare All mode** — side-by-side view of all strategies at once
- **Stats dashboard** — chunk count, average/min/max sizes per strategy
- **No LLM needed** — pure chunking visualization, lightweight and fast

## ✂️ Chunking Strategies

| # | Strategy | How It Works | Best For |
|---|---|---|---|
| 1 | 📏 **Fixed-size** | Equal character count with overlap | File storage, streaming, ML batching |
| 2 | 📐 **Variable-size** | Sentence boundaries, min/max range | Deduplication, irregular data |
| 3 | 🔍 **Content-based** | Split on headings, sections, separators | Backup systems, structured docs |
| 4 | 🧩 **Logical** | By paragraphs as logical units | Text analysis, NLP preprocessing |
| 5 | ⚡ **Dynamic** | Adaptive sizing based on content density | Real-time analytics, streaming |
| 6 | 📄 **File-based** | One chunk per PDF page | Cloud storage, video streaming |
| 7 | ⚙️ **Task-based** | Balanced across N parallel workers | Distributed computing, ML training |

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| PDF Parsing | pypdf |
| UI | Vanilla HTML/CSS/JS |

> **Note:** This project does **not** require Ollama, ChromaDB, or any ML models. It's a lightweight visualization tool.

## 📁 Project Structure

```
rag_chunking/
├── main.py              ← FastAPI app + 7 chunking implementations
├── templates/
│   └── index.html       ← Visual comparison UI
├── requirements.txt     ← Minimal Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- No LLM or external services needed!

### Installation

```bash
# Navigate to the project
cd rag_chunking

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --port 8003
```

Open **http://localhost:8003** in your browser.

### Usage

1. **Upload** — Drag & drop a PDF file
2. **Browse strategies** — Click tabs to switch between the 7 methods
3. **Adjust parameters** — Modify chunk size, overlap, workers, etc.
4. **Run** — Click "Run Chunking" to see color-coded chunk cards
5. **Compare** — Switch to "Compare All" mode for a side-by-side view

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/upload` | Upload a PDF, returns extracted text |
| `POST` | `/api/chunk` | Apply a strategy (JSON body with params) |
| `GET` | `/api/strategies` | List all available strategies |

### Chunk Request Body

```json
{
  "strategy": "fixed_size",
  "text": "...",
  "pages": ["page1 text", "page2 text"],
  "chunk_size": 500,
  "chunk_overlap": 50,
  "min_size": 200,
  "max_size": 800,
  "num_workers": 4
}
```

## 📊 Understanding the Output

Each chunk card shows:
- **Chunk index** — color-coded badge
- **Character count** — total characters in the chunk
- **Word count** — total words in the chunk
- **Size bar** — visual representation relative to the largest chunk
- **Text preview** — expandable content preview
