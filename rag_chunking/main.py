"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           RAG Chunking Demo — Visualize 7 Chunking Strategies              ║
║                        Workshop Edition v2.0                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Upload a PDF and SEE how each chunking strategy splits the same text.     ║
║                                                                            ║
║  Strategies:                                                               ║
║    1. Fixed-size     — Equal character-count chunks                        ║
║    2. Variable-size  — Sentence/paragraph boundaries with size range       ║
║    3. Content-based  — Split on headings, sections, patterns               ║
║    4. Logical        — By paragraphs and sentences as logical units        ║
║    5. Dynamic        — Adaptive sizing based on content density            ║
║    6. File-based     — Split by PDF page boundaries                        ║
║    7. Task-based     — Balanced chunks for N parallel workers              ║
║                                                                            ║
║  Run:  uvicorn main:app --reload --port 8003                               ║
║  Open: http://localhost:8003                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import io
import re
import math
from typing import List, Optional
from contextlib import asynccontextmanager

import pypdf

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.requests import Request
from pydantic import BaseModel


# =============================================================================
# SECTION 2: DATA MODELS
# =============================================================================

class ChunkRequest(BaseModel):
    """Request body for the /api/chunk endpoint."""
    strategy: str          # One of the 7 strategy names
    text: str              # The full document text
    pages: Optional[List[str]] = None  # Per-page text (for file-based chunking)
    # Strategy-specific parameters
    chunk_size: int = 500
    chunk_overlap: int = 50
    min_size: int = 200
    max_size: int = 800
    num_workers: int = 4


class ChunkResult(BaseModel):
    """A single chunk with metadata."""
    index: int
    text: str
    char_count: int
    word_count: int


# =============================================================================
# SECTION 3: CHUNKING STRATEGIES
# =============================================================================

def chunk_fixed_size(text: str, chunk_size: int = 500, overlap: int = 50) -> List[ChunkResult]:
    """
    ── STRATEGY 1: FIXED-SIZE CHUNKING ──
    Divides text into equal-sized character chunks with optional overlap.

    How it works:
    - Walk through the text in steps of (chunk_size - overlap)
    - Each chunk is exactly chunk_size characters (except possibly the last one)
    - Overlap ensures no information is lost at chunk boundaries

    Use cases: file storage, streaming data, ML batch processing
    """
    chunks = []
    step = max(1, chunk_size - overlap)
    i = 0
    idx = 0
    while i < len(text):
        chunk_text = text[i : i + chunk_size]
        if chunk_text.strip():
            chunks.append(ChunkResult(
                index=idx,
                text=chunk_text,
                char_count=len(chunk_text),
                word_count=len(chunk_text.split()),
            ))
            idx += 1
        i += step
    return chunks


def chunk_variable_size(text: str, min_size: int = 200, max_size: int = 800) -> List[ChunkResult]:
    """
    ── STRATEGY 2: VARIABLE-SIZE CHUNKING ──
    Splits on sentence boundaries, accumulating until within size range.

    How it works:
    - Split text into sentences
    - Accumulate sentences until the chunk is within [min_size, max_size]
    - When adding another sentence would exceed max_size, close the current chunk
    - Results in chunks of varying sizes that respect sentence boundaries

    Use cases: deduplication, handling irregular data patterns
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current = ""
    idx = 0

    for sentence in sentences:
        if current and len(current) + len(sentence) > max_size:
            if current.strip():
                chunks.append(ChunkResult(
                    index=idx, text=current.strip(),
                    char_count=len(current.strip()),
                    word_count=len(current.strip().split()),
                ))
                idx += 1
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence

    if current.strip():
        # If last chunk is too small, merge with previous
        if len(current.strip()) < min_size and chunks:
            prev = chunks[-1]
            merged = prev.text + " " + current.strip()
            chunks[-1] = ChunkResult(
                index=prev.index, text=merged,
                char_count=len(merged), word_count=len(merged.split()),
            )
        else:
            chunks.append(ChunkResult(
                index=idx, text=current.strip(),
                char_count=len(current.strip()),
                word_count=len(current.strip().split()),
            ))

    return chunks


def chunk_content_based(text: str) -> List[ChunkResult]:
    """
    ── STRATEGY 3: CONTENT-BASED CHUNKING ──
    Splits on content patterns: headings, section markers, horizontal rules.

    How it works:
    - Look for patterns that indicate section boundaries:
      • Markdown headings (# / ## / ###)
      • ALL-CAPS headings on their own line
      • Separator lines (---, ===, ***)
      • Numbered section headers (1. / 1.1 / Section X)
    - Split text at these boundaries
    - Each resulting chunk is a content section

    Use cases: backup systems, deduplication, document sections
    """
    # Patterns that indicate section boundaries
    pattern = r'(?=(?:^|\n)(?:#{1,4}\s|[A-Z][A-Z\s]{4,}$|[-=*]{3,}|(?:\d+\.)+\s|Section\s+\d|CHAPTER\s))'
    sections = re.split(pattern, text)

    chunks = []
    idx = 0
    for section in sections:
        section = section.strip()
        if section and len(section) > 20:  # Skip tiny fragments
            chunks.append(ChunkResult(
                index=idx, text=section,
                char_count=len(section),
                word_count=len(section.split()),
            ))
            idx += 1

    # If no patterns found, fall back to paragraph splitting
    if len(chunks) <= 1:
        return chunk_logical(text)

    return chunks


def chunk_logical(text: str) -> List[ChunkResult]:
    """
    ── STRATEGY 4: LOGICAL CHUNKING ──
    Splits by logical units: paragraphs (double newlines).

    How it works:
    - Split on double newlines (paragraph boundaries)
    - Each paragraph becomes its own chunk
    - Very short paragraphs (< 50 chars) are merged with the next one
    - Preserves the author's intended text structure

    Use cases: text analysis, NLP preprocessing, document understanding
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    idx = 0
    buffer = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        buffer = (buffer + "\n\n" + para).strip() if buffer else para

        # Only emit if the paragraph is substantial enough
        if len(buffer) >= 50:
            chunks.append(ChunkResult(
                index=idx, text=buffer,
                char_count=len(buffer),
                word_count=len(buffer.split()),
            ))
            idx += 1
            buffer = ""

    # Emit any remaining buffer
    if buffer.strip():
        if chunks:
            prev = chunks[-1]
            merged = prev.text + "\n\n" + buffer.strip()
            chunks[-1] = ChunkResult(
                index=prev.index, text=merged,
                char_count=len(merged), word_count=len(merged.split()),
            )
        else:
            chunks.append(ChunkResult(
                index=idx, text=buffer.strip(),
                char_count=len(buffer.strip()),
                word_count=len(buffer.strip().split()),
            ))

    return chunks


def chunk_dynamic(text: str, min_size: int = 200, max_size: int = 800) -> List[ChunkResult]:
    """
    ── STRATEGY 5: DYNAMIC CHUNKING ──
    Adaptive sizing based on content density (punctuation, whitespace).

    How it works:
    - Analyze each paragraph's "density" (ratio of punctuation + digits to total chars)
    - Dense content (lots of data, numbers, lists) → smaller chunks for precision
    - Sparse content (narrative prose) → larger chunks for context
    - Target size = min_size for densest content, max_size for sparsest
    - Accumulate sentences until target is reached

    Use cases: streaming applications, real-time analytics, adaptive systems
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para or len(para) < 30:
            continue

        # Calculate content density
        special_chars = sum(1 for c in para if c in '.,;:!?()[]{}0123456789%$#@&*')
        density = special_chars / max(len(para), 1)

        # Denser content → smaller target; sparse → larger target
        target_size = int(max_size - (max_size - min_size) * min(density * 5, 1.0))

        # If paragraph is within target, emit as one chunk
        if len(para) <= target_size:
            chunks.append(ChunkResult(
                index=idx, text=para,
                char_count=len(para),
                word_count=len(para.split()),
            ))
            idx += 1
        else:
            # Split large paragraphs by sentences, respecting target
            sentences = re.split(r'(?<=[.!?])\s+', para)
            current = ""
            for sent in sentences:
                if current and len(current) + len(sent) > target_size:
                    chunks.append(ChunkResult(
                        index=idx, text=current.strip(),
                        char_count=len(current.strip()),
                        word_count=len(current.strip().split()),
                    ))
                    idx += 1
                    current = sent
                else:
                    current = (current + " " + sent).strip() if current else sent
            if current.strip():
                chunks.append(ChunkResult(
                    index=idx, text=current.strip(),
                    char_count=len(current.strip()),
                    word_count=len(current.strip().split()),
                ))
                idx += 1

    return chunks


def chunk_file_based(pages: List[str]) -> List[ChunkResult]:
    """
    ── STRATEGY 6: FILE-BASED CHUNKING ──
    Splits by PDF page boundaries.

    How it works:
    - Each PDF page becomes exactly one chunk
    - Preserves the document's physical layout
    - Page number is stored in the chunk index

    Use cases: file-sharing systems, cloud storage, video streaming segments
    """
    chunks = []
    for i, page_text in enumerate(pages):
        page_text = page_text.strip()
        if page_text:
            chunks.append(ChunkResult(
                index=i,
                text=page_text,
                char_count=len(page_text),
                word_count=len(page_text.split()),
            ))
    return chunks


def chunk_task_based(text: str, num_workers: int = 4) -> List[ChunkResult]:
    """
    ── STRATEGY 7: TASK-BASED CHUNKING ──
    Balanced chunks optimized for N parallel workers.

    How it works:
    - Split text into paragraphs first
    - Use a greedy bin-packing algorithm to distribute paragraphs
      across N workers as evenly as possible (by character count)
    - Each worker's assigned paragraphs become one chunk
    - Result: N roughly equal-sized chunks

    Use cases: parallel ML training, distributed computing, map-reduce
    """
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    if not paragraphs:
        return []

    # Ensure we don't have more workers than paragraphs
    n = min(num_workers, len(paragraphs))

    # Initialize bins (one per worker)
    bins: List[List[str]] = [[] for _ in range(n)]
    bin_sizes = [0] * n

    # Greedy bin-packing: assign each paragraph to the lightest bin
    for para in paragraphs:
        lightest = bin_sizes.index(min(bin_sizes))
        bins[lightest].append(para)
        bin_sizes[lightest] += len(para)

    chunks = []
    for i, bin_paragraphs in enumerate(bins):
        combined = "\n\n".join(bin_paragraphs)
        if combined.strip():
            chunks.append(ChunkResult(
                index=i,
                text=combined,
                char_count=len(combined),
                word_count=len(combined.split()),
            ))

    return chunks


# Strategy registry
STRATEGIES = {
    "fixed_size": {
        "name": "Fixed-size Chunking",
        "func": lambda req: chunk_fixed_size(req.text, req.chunk_size, req.chunk_overlap),
        "description": "Divides text into equal-sized character chunks with overlap. Straightforward and ideal for file storage, streaming data, and ML batching.",
        "icon": "📏",
    },
    "variable_size": {
        "name": "Variable-size Chunking",
        "func": lambda req: chunk_variable_size(req.text, req.min_size, req.max_size),
        "description": "Splits on sentence boundaries, creating chunks of varying sizes within a min/max range. Ideal for deduplication and irregular data patterns.",
        "icon": "📐",
    },
    "content_based": {
        "name": "Content-based Chunking",
        "func": lambda req: chunk_content_based(req.text),
        "description": "Splits on content patterns like headings, section markers, and separators. Used for backup and deduplication systems with structured content.",
        "icon": "🔍",
    },
    "logical": {
        "name": "Logical Chunking",
        "func": lambda req: chunk_logical(req.text),
        "description": "Breaks text by logical units — paragraphs and sentences — preserving the author's intended structure.",
        "icon": "🧩",
    },
    "dynamic": {
        "name": "Dynamic Chunking",
        "func": lambda req: chunk_dynamic(req.text, req.min_size, req.max_size),
        "description": "Adaptive sizing based on content density. Dense content (numbers, lists) → smaller chunks; sparse prose → larger chunks. Ideal for streaming and real-time analytics.",
        "icon": "⚡",
    },
    "file_based": {
        "name": "File-based Chunking",
        "func": lambda req: chunk_file_based(req.pages or [req.text]),
        "description": "Splits by PDF page boundaries. Each page becomes one chunk, preserving the physical document layout. Used for file-sharing and cloud storage.",
        "icon": "📄",
    },
    "task_based": {
        "name": "Task-based Chunking",
        "func": lambda req: chunk_task_based(req.text, req.num_workers),
        "description": "Balanced chunks optimized for N parallel workers using greedy bin-packing. Used for parallel ML training and distributed systems.",
        "icon": "⚙️",
    },
}


# =============================================================================
# SECTION 4: FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("✂️  Starting RAG Chunking Demo")
    print("=" * 60)
    print("✨ Ready! Open http://localhost:8003")
    print("=" * 60 + "\n")
    yield
    print("\n👋 Shutting down Chunking Demo...")


app = FastAPI(
    title="RAG Chunking Demo",
    description="Visualize 7 chunking strategies on your documents.",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# =============================================================================
# SECTION 5: API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the chunking visualization UI."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a PDF and extract text (full + per-page).

    Returns:
        - text: Full concatenated text
        - pages: List of per-page text strings
        - page_count: Number of pages
        - total_chars: Total character count
    """
    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        file_bytes = await file.read()
        pdf_file = io.BytesIO(file_bytes)
        reader = pypdf.PdfReader(pdf_file)

        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)

        full_text = "\n\n".join(pages)

        print(f"📥 Uploaded: {file.filename} ({len(pages)} pages, {len(full_text)} chars)")

        return {
            "filename": file.filename,
            "text": full_text,
            "pages": pages,
            "page_count": len(pages),
            "total_chars": len(full_text),
            "total_words": len(full_text.split()),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.post("/api/chunk")
async def chunk_document(request: ChunkRequest):
    """
    Apply a chunking strategy to the given text.

    Returns chunks with stats for visualization.
    """
    if request.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. Available: {list(STRATEGIES.keys())}",
        )

    strategy = STRATEGIES[request.strategy]

    try:
        chunks = strategy["func"](request)

        char_counts = [c.char_count for c in chunks]
        word_counts = [c.word_count for c in chunks]

        stats = {
            "strategy": request.strategy,
            "strategy_name": strategy["name"],
            "total_chunks": len(chunks),
            "avg_chars": round(sum(char_counts) / len(char_counts)) if char_counts else 0,
            "min_chars": min(char_counts) if char_counts else 0,
            "max_chars": max(char_counts) if char_counts else 0,
            "avg_words": round(sum(word_counts) / len(word_counts)) if word_counts else 0,
        }

        print(f"  ✂️  {strategy['name']}: {len(chunks)} chunks (avg {stats['avg_chars']} chars)")

        return {
            "chunks": [c.dict() for c in chunks],
            "stats": stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")


@app.get("/api/strategies")
async def list_strategies():
    """Returns metadata about all available chunking strategies."""
    return {
        key: {
            "name": val["name"],
            "description": val["description"],
            "icon": val["icon"],
        }
        for key, val in STRATEGIES.items()
    }


# =============================================================================
# Run: uvicorn main:app --reload --port 8003
# Open: http://localhost:8003
# =============================================================================
