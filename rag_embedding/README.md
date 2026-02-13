# 🔢 RAG Embedding Demo — Visualize 5 Methods

An interactive tool to **visualize and compare** how 5 different embedding methods represent the same sentences. See heatmaps, similarity matrices, 2D cluster plots, and vocabulary.

---

## ✨ Features

- **5 embedding methods** — from basic one-hot to state-of-the-art sentence transformers
- **Embedding heatmap** — color grid showing vector values per sentence
- **Cosine similarity matrix** — how similar sentences are to each other per method
- **2D PCA scatter plot** — see how sentences cluster in 2D space
- **Compare All mode** — side-by-side comparison of all 5 methods with mini plots
- **Preset sentence sets** — animals, tech, semantic similarity, mixed topics
- **Vocabulary inspector** — see the words/dimensions each method uses

## 🔢 Embedding Methods

| # | Method | Type | Dimensions | How It Works |
|---|---|---|---|---|
| 1 | 🔤 **One-Hot** | Sparse | vocab_size | Binary: 1 at each word's position |
| 2 | 🎒 **Bag of Words** | Sparse | vocab_size | Word frequency counts |
| 3 | ⚖️ **TF-IDF** | Sparse | vocab_size | Frequency × rarity weighting |
| 4 | 🧠 **Word2Vec** | Dense | 50 | Averaged word vectors (trained on input) |
| 5 | 🚀 **Sentence Trans.** | Dense | 384 | Pre-trained `all-MiniLM-L6-v2` |

## 🏗️ Tech Stack

| Component | Technology |
|---|---|
| Web Framework | FastAPI |
| Sparse Methods | scikit-learn (CountVectorizer, TfidfVectorizer) |
| Word2Vec | gensim |
| Sentence Embeddings | sentence-transformers |
| Dimensionality Reduction | PCA (scikit-learn) |
| Similarity | Cosine similarity (scikit-learn) |

## 📁 Project Structure

```
rag_embedding/
├── main.py              ← FastAPI app + 5 embedding implementations
├── templates/
│   └── index.html       ← Visual comparison UI
├── requirements.txt     ← Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- No LLM or Ollama needed!

### Installation

```bash
# Navigate to the project
cd rag_embedding

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # macOS/Linux
# venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
uvicorn main:app --reload --port 8004
```

Open **http://localhost:8004** in your browser.

### Usage

1. **Enter sentences** — Type or paste sentences (one per line), or use a preset
2. **Select method** — Click a method tab to embed with that method
3. **Explore visualizations:**
   - 🗺️ **Heatmap** — see the raw embedding vector values
   - 🔗 **Similarity** — cosine similarity between all sentence pairs
   - 📍 **2D Plot** — PCA projection showing sentence clusters
   - 📚 **Vocabulary** — words used as dimensions (sparse methods)
4. **Compare All** — click to see all 5 methods side-by-side

## 🔧 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/api/embed` | Embed with one method (JSON: `{sentences, method}`) |
| `POST` | `/api/embed_all` | Embed with all methods (JSON: `{sentences}`) |
| `GET` | `/api/methods` | List available methods |

## 📊 Understanding the Visualizations

### Heatmap
- Each row = one sentence, each column = one dimension
- 🟠 Orange = positive values, 🔵 Blue = negative values
- Brighter = larger magnitude

### Similarity Matrix
- Each cell = cosine similarity between two sentences (0 to 1)
- 🟢 Bright green = very similar, ⚫ Dark = dissimilar
- Diagonal is always 1.0 (self-similarity)

### 2D Plot
- Each dot = one sentence projected to 2D via PCA
- Close dots = semantically similar sentences
- Axis labels show % variance explained
