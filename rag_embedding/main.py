"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RAG Embedding Demo — Visualize 5 Embedding Methods               ║
║                        Workshop Edition v2.0                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Enter sentences and SEE how each embedding method represents them.        ║
║                                                                            ║
║  Methods:                                                                  ║
║    1. One-Hot Encoding  — Binary word-position vectors                     ║
║    2. Bag of Words      — Word frequency counts                            ║
║    3. TF-IDF            — Frequency × rarity weighting                     ║
║    4. Word2Vec          — Averaged dense word vectors                      ║
║    5. Sentence Trans.   — Pre-trained 384-dim sentence embeddings          ║
║                                                                            ║
║  Visualizations:                                                           ║
║    • Embedding heatmap  — color grid of vector values                      ║
║    • Similarity matrix  — cosine similarity between sentences              ║
║    • 2D scatter plot    — PCA projection of embeddings                     ║
║                                                                            ║
║  Run:  uvicorn main:app --reload --port 8004                               ║
║  Open: http://localhost:8004                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import re
import math
import numpy as np
from typing import List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer


# =============================================================================
# SECTION 2: DATA MODELS
# =============================================================================

class EmbedRequest(BaseModel):
    """Request body for embedding endpoints."""
    sentences: List[str]
    method: str  # one_hot, bow, tfidf, word2vec, sentence_transformer


class EmbedAllRequest(BaseModel):
    """Request body for the compare-all endpoint."""
    sentences: List[str]


# =============================================================================
# SECTION 3: EMBEDDING METHODS
# =============================================================================

def tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r'\b\w+\b', text.lower())


def embed_one_hot(sentences: List[str]) -> Dict:
    """
    ── METHOD 1: ONE-HOT ENCODING ──
    Each unique word in the vocabulary gets its own dimension.
    A sentence's vector has 1 at the position of each word it contains, 0 elsewhere.

    Properties:
    - Very sparse (mostly zeros)
    - No notion of word similarity
    - Dimension = vocabulary size
    - Simple and interpretable

    Use cases: simple classification, feature indexing
    """
    all_tokens = []
    tokenized = []
    for sent in sentences:
        tokens = tokenize(sent)
        tokenized.append(tokens)
        all_tokens.extend(tokens)

    vocab = sorted(set(all_tokens))
    vocab_idx = {w: i for i, w in enumerate(vocab)}

    vectors = []
    for tokens in tokenized:
        vec = [0.0] * len(vocab)
        for token in tokens:
            vec[vocab_idx[token]] = 1.0
        vectors.append(vec)

    return {
        "vectors": vectors,
        "dimensions": len(vocab),
        "vocab": vocab,
        "method_type": "sparse",
        "description": "Binary vector: 1 at each word's position, 0 elsewhere. No frequency info, no word similarity.",
    }


def embed_bow(sentences: List[str]) -> Dict:
    """
    ── METHOD 2: BAG OF WORDS ──
    Counts how many times each word appears in a sentence.

    Properties:
    - Sparse (most entries are 0)
    - Captures word frequency
    - Ignores word order (hence "bag")
    - Dimension = vocabulary size

    Use cases: text classification, spam detection
    """
    vectorizer = CountVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    vectors = matrix.toarray().tolist()
    vocab = vectorizer.get_feature_names_out().tolist()

    return {
        "vectors": vectors,
        "dimensions": len(vocab),
        "vocab": vocab,
        "method_type": "sparse",
        "description": "Word frequency counts per sentence. Captures word importance by count but ignores order.",
    }


def embed_tfidf(sentences: List[str]) -> Dict:
    """
    ── METHOD 3: TF-IDF ──
    Term Frequency × Inverse Document Frequency.

    TF  = how often a word appears in THIS sentence
    IDF = how rare a word is across ALL sentences

    Properties:
    - Sparse, but with float values (0.0 – 1.0)
    - Common words (the, is, a) get low scores
    - Rare, distinctive words get high scores
    - Dimension = vocabulary size

    Use cases: search engines, document retrieval, keyword extraction
    """
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(sentences)
    vectors = matrix.toarray().tolist()
    vocab = vectorizer.get_feature_names_out().tolist()
    # Round for cleaner display
    vectors = [[round(v, 4) for v in vec] for vec in vectors]

    return {
        "vectors": vectors,
        "dimensions": len(vocab),
        "vocab": vocab,
        "method_type": "sparse",
        "description": "TF (word frequency) × IDF (word rarity). Rare distinctive words score high; common words score low.",
    }


def embed_word2vec(sentences: List[str]) -> Dict:
    """
    ── METHOD 4: WORD2VEC (AVERAGED) ──
    Trains word-level embeddings, then averages all word vectors in a sentence.

    Properties:
    - Dense (all values are non-zero floats)
    - Captures semantic similarity (king - man + woman ≈ queen)
    - Fixed dimension (50-dim in this demo)
    - Trained on the input text (small corpus → limited quality)

    Use cases: word similarity, analogy tasks, feature extraction
    """
    dim = 50
    tokenized = [tokenize(sent) for sent in sentences]

    # Train Word2Vec on the input sentences
    model = Word2Vec(
        sentences=tokenized,
        vector_size=dim,
        window=5,
        min_count=1,
        workers=1,
        epochs=100,
        seed=42,
    )

    vectors = []
    for tokens in tokenized:
        word_vecs = [model.wv[w].tolist() for w in tokens if w in model.wv]
        if word_vecs:
            avg = [round(sum(col) / len(col), 4) for col in zip(*word_vecs)]
        else:
            avg = [0.0] * dim
        vectors.append(avg)

    return {
        "vectors": vectors,
        "dimensions": dim,
        "vocab": list(model.wv.key_to_index.keys())[:30],
        "method_type": "dense",
        "description": "Dense word vectors (50-dim) trained on input text, averaged per sentence. Captures semantic meaning.",
    }


# Global sentence transformer model (loaded at startup)
st_model = None


def embed_sentence_transformer(sentences: List[str]) -> Dict:
    """
    ── METHOD 5: SENTENCE TRANSFORMERS ──
    Pre-trained model that embeds entire sentences into fixed-size dense vectors.

    Properties:
    - Dense (384 dimensions, all non-zero)
    - Captures full sentence semantics, not just words
    - Pre-trained on billions of text pairs
    - State-of-the-art for semantic similarity

    Use cases: semantic search, RAG retrieval, clustering, deduplication
    """
    embeddings = st_model.encode(sentences)
    vectors = [[round(float(v), 4) for v in vec] for vec in embeddings]

    return {
        "vectors": vectors,
        "dimensions": len(vectors[0]) if vectors else 0,
        "vocab": [],
        "method_type": "dense",
        "description": "Pre-trained 384-dim sentence embeddings (all-MiniLM-L6-v2). State-of-the-art semantic understanding.",
    }


# Strategy registry
METHODS = {
    "one_hot": {
        "name": "One-Hot Encoding",
        "func": embed_one_hot,
        "icon": "🔤",
        "description": "Binary vector: 1 at each word's position. Simplest representation — no word similarity, no frequency info.",
        "type": "Sparse",
    },
    "bow": {
        "name": "Bag of Words",
        "func": embed_bow,
        "icon": "🎒",
        "description": "Word frequency counts per sentence. Captures how often each word appears but ignores word order entirely.",
        "type": "Sparse",
    },
    "tfidf": {
        "name": "TF-IDF",
        "func": embed_tfidf,
        "icon": "⚖️",
        "description": "Term Frequency × Inverse Document Frequency. Common words score low, rare distinctive words score high.",
        "type": "Sparse",
    },
    "word2vec": {
        "name": "Word2Vec (Averaged)",
        "func": embed_word2vec,
        "icon": "🧠",
        "description": "Dense word vectors trained on the input, averaged per sentence. Captures semantic relationships between words.",
        "type": "Dense",
    },
    "sentence_transformer": {
        "name": "Sentence Transformers",
        "func": embed_sentence_transformer,
        "icon": "🚀",
        "description": "Pre-trained 384-dim model (all-MiniLM-L6-v2). State-of-the-art semantic understanding of full sentences.",
        "type": "Dense",
    },
}


# =============================================================================
# SECTION 4: VISUALIZATION HELPERS
# =============================================================================

def compute_similarity_matrix(vectors: List[List[float]]) -> List[List[float]]:
    """Compute pairwise cosine similarity between all sentence vectors."""
    arr = np.array(vectors)
    if arr.shape[0] < 2:
        return [[1.0]]
    sim = cosine_similarity(arr)
    return [[round(float(v), 4) for v in row] for row in sim]


def compute_pca_2d(vectors: List[List[float]]) -> List[Dict]:
    """Project vectors to 2D using PCA for scatter plot visualization."""
    arr = np.array(vectors)
    if arr.shape[0] < 2:
        return [{"x": 0.0, "y": 0.0}]
    n_components = min(2, arr.shape[0], arr.shape[1])
    pca = PCA(n_components=n_components)
    projected = pca.fit_transform(arr)
    variance = [round(float(v) * 100, 1) for v in pca.explained_variance_ratio_]
    result = []
    for point in projected:
        x = round(float(point[0]), 4)
        y = round(float(point[1]), 4) if len(point) > 1 else 0.0
        result.append({"x": x, "y": y})
    return result, variance


def compute_stats(vectors: List[List[float]]) -> Dict:
    """Compute statistics about the embedding vectors."""
    arr = np.array(vectors)
    nonzero = np.count_nonzero(arr)
    total = arr.size
    sparsity = round(1.0 - (nonzero / total), 4) if total > 0 else 0
    magnitudes = [round(float(np.linalg.norm(vec)), 4) for vec in arr]
    return {
        "sparsity": sparsity,
        "avg_magnitude": round(float(np.mean(magnitudes)), 4),
        "min_magnitude": round(float(np.min(magnitudes)), 4),
        "max_magnitude": round(float(np.max(magnitudes)), 4),
    }


# =============================================================================
# SECTION 5: FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global st_model
    print("=" * 60)
    print("🔢 Starting RAG Embedding Demo")
    print("=" * 60)

    print("\n📦 Loading Sentence Transformer model...")
    st_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("   ✅ Model loaded!")

    print("\n" + "=" * 60)
    print("✨ Ready! Open http://localhost:8004")
    print("=" * 60 + "\n")

    yield
    print("\n👋 Shutting down Embedding Demo...")


app = FastAPI(
    title="RAG Embedding Demo",
    description="Visualize and compare 5 embedding methods.",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# =============================================================================
# SECTION 6: API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the embedding visualization UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/embed")
async def embed_sentences(request: EmbedRequest):
    """
    Embed sentences using a specific method.

    Returns vectors, similarity matrix, 2D PCA projection, and stats.
    """
    if not request.sentences or all(not s.strip() for s in request.sentences):
        raise HTTPException(status_code=400, detail="Please provide at least one sentence.")

    sentences = [s.strip() for s in request.sentences if s.strip()]

    if request.method not in METHODS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown method: {request.method}. Available: {list(METHODS.keys())}",
        )

    method_info = METHODS[request.method]

    try:
        print(f"  🔢 {method_info['name']}: embedding {len(sentences)} sentences...")
        result = method_info["func"](sentences)

        # Compute visualizations
        similarity = compute_similarity_matrix(result["vectors"])
        pca_points, variance = compute_pca_2d(result["vectors"])
        stats = compute_stats(result["vectors"])

        # Truncate vectors for display if too large
        display_vectors = result["vectors"]
        truncated = False
        max_display_dims = 60
        if result["dimensions"] > max_display_dims:
            display_vectors = [vec[:max_display_dims] for vec in result["vectors"]]
            truncated = True

        return {
            "method": request.method,
            "method_name": method_info["name"],
            "method_type": result["method_type"],
            "description": result["description"],
            "sentences": sentences,
            "vectors": display_vectors,
            "full_dimensions": result["dimensions"],
            "display_dimensions": min(result["dimensions"], max_display_dims),
            "truncated": truncated,
            "vocab": result.get("vocab", [])[:40],
            "similarity_matrix": similarity,
            "pca_2d": pca_points,
            "pca_variance": variance,
            "stats": stats,
        }

    except Exception as e:
        print(f"  ❌ Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/api/embed_all")
async def embed_all_methods(request: EmbedAllRequest):
    """
    Embed sentences with ALL methods at once for comparison.
    Returns a summary per method (dimensions, sparsity, similarity range).
    """
    sentences = [s.strip() for s in request.sentences if s.strip()]
    if not sentences:
        raise HTTPException(status_code=400, detail="Please provide at least one sentence.")

    results = {}
    for key, info in METHODS.items():
        try:
            result = info["func"](sentences)
            similarity = compute_similarity_matrix(result["vectors"])
            pca_points, variance = compute_pca_2d(result["vectors"])
            stats = compute_stats(result["vectors"])

            # Flatten sim matrix (exclude diagonal)
            sim_values = []
            for i, row in enumerate(similarity):
                for j, v in enumerate(row):
                    if i != j:
                        sim_values.append(v)

            results[key] = {
                "name": info["name"],
                "icon": info["icon"],
                "type": info["type"],
                "dimensions": result["dimensions"],
                "sparsity": stats["sparsity"],
                "avg_magnitude": stats["avg_magnitude"],
                "sim_min": round(min(sim_values), 4) if sim_values else 0,
                "sim_max": round(max(sim_values), 4) if sim_values else 0,
                "sim_avg": round(sum(sim_values) / len(sim_values), 4) if sim_values else 0,
                "pca_2d": pca_points,
                "pca_variance": variance,
                "similarity_matrix": similarity,
            }
        except Exception as e:
            results[key] = {"name": info["name"], "error": str(e)}

    return {"sentences": sentences, "results": results}


@app.get("/api/methods")
async def list_methods():
    """Returns metadata about all available embedding methods."""
    return {
        key: {
            "name": val["name"],
            "icon": val["icon"],
            "description": val["description"],
            "type": val["type"],
        }
        for key, val in METHODS.items()
    }


# =============================================================================
# Run: uvicorn main:app --reload --port 8004
# Open: http://localhost:8004
# =============================================================================
