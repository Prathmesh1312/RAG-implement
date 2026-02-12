"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              RAG Online — Web Link Ingestion Demo                          ║
║                     Workshop Edition v2.0                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This project handles ONLINE link ingestion only.                          ║
║                                                                            ║
║  Supported source types:                                                   ║
║    • Web PDFs    — PDF files hosted at a URL                               ║
║    • Web Articles— HTML pages (Medium blogs, news, etc.)                   ║
║    • Local paths — PDF files on disk (via path string)                     ║
║                                                                            ║
║  Pipeline:                                                                 ║
║  1. DETECT  → Classify source type (web_pdf / web_article / local_pdf)     ║
║  2. EXTRACT → Download and extract text                                    ║
║  3. CHUNK   → Split text into overlapping chunks                           ║
║  4. EMBED   → Convert chunks into vector embeddings                        ║
║  5. STORE   → Persist embeddings in ChromaDB                               ║
║  6. QUERY   → Retrieve relevant chunks and generate answers via LLM        ║
║                                                                            ║
║  Run:  uvicorn main:app --reload --port 8002                               ║
║  Open: http://localhost:8002                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os                          # File system operations
import io                          # In-memory binary streams
from urllib.parse import urlparse  # URL parsing
from typing import List            # Type hints
from contextlib import asynccontextmanager

import requests                    # HTTP client for downloading
import pypdf                       # PDF text extraction
from bs4 import BeautifulSoup      # HTML scraping

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.templating import Jinja2Templates
from starlette.requests import Request

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# =============================================================================
# SECTION 2: CONFIGURATION
# =============================================================================

CHROMA_PERSIST_DIR = "./chroma_db_online"          # Separate DB for online project
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "gemma3:4b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RESULTS = 10


# =============================================================================
# SECTION 3: SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """
You are an expert document analyst. Your task is to answer questions based SOLELY
on the provided document context.

Rules:
1. Do NOT use any outside knowledge — only use the context provided below.
2. If the answer is not available in the context, clearly state that you cannot
   find the answer in the provided documents.
3. Be concise and direct. Include all key details: numbers, percentages, dates,
   and specific conditions mentioned in the text.
4. Do NOT include your reasoning process in the answer.

Context:
{context}
"""


# =============================================================================
# SECTION 4: SOURCE DETECTION & TEXT EXTRACTION
# =============================================================================

def detect_source_type(source: str) -> str:
    """
    Classifies a source string as:
    • 'web_pdf'     → URL pointing to a PDF
    • 'web_article' → URL pointing to an HTML page
    • 'local_pdf'   → Local file path
    """
    if source.startswith("http://") or source.startswith("https://"):
        parsed = urlparse(source)
        if parsed.path.lower().endswith(".pdf"):
            return "web_pdf"
        return "web_article"
    else:
        return "local_pdf"


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extracts text from raw PDF bytes using pypdf."""
    pdf_file = io.BytesIO(pdf_bytes)
    reader = pypdf.PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text


def extract_text_from_local_pdf(file_path: str) -> str:
    """Reads a local PDF file and extracts its text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF file not found: {file_path}")
    with open(file_path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def extract_text_from_web_pdf(url: str) -> str:
    """Downloads a PDF from a URL and extracts its text."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return extract_text_from_pdf_bytes(response.content)


def scrape_web_article(url: str) -> str:
    """
    Scrapes readable text from a web article.

    Strategy:
    1. Send GET with browser User-Agent
    2. Parse HTML with BeautifulSoup
    3. Remove noise (scripts, nav, etc.)
    4. Try <article> tag first, fall back to all <p> tags
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    article = soup.find("article")
    if article:
        return article.get_text(separator="\n", strip=True)

    paragraphs = soup.find_all("p")
    return "\n".join(p.get_text(strip=True) for p in paragraphs)


def extract_text(source: str) -> str:
    """
    Unified extraction entry point — detects source type and routes
    to the appropriate extractor.
    """
    source_type = detect_source_type(source)
    print(f"  📄 Source type detected: {source_type}")

    if source_type == "local_pdf":
        return extract_text_from_local_pdf(source)
    elif source_type == "web_pdf":
        return extract_text_from_web_pdf(source)
    elif source_type == "web_article":
        return scrape_web_article(source)
    else:
        raise ValueError(f"Unknown source type: {source_type}")


# =============================================================================
# SECTION 5: VECTOR STORE & RAG CHAIN
# =============================================================================

def chunk_text(text: str, source_name: str) -> List[Document]:
    """Splits text into overlapping chunks for retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    doc = Document(page_content=text, metadata={"source": source_name})
    chunks = text_splitter.split_documents([doc])
    print(f"  ✂️  Split into {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def get_or_create_vector_store(embeddings) -> Chroma:
    """Initializes or loads the ChromaDB vector store."""
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


def format_docs(docs: List[Document]) -> str:
    """Joins retrieved document chunks into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store: Chroma, llm):
    """
    Assembles the RAG chain using LCEL.

    Pipeline:
        question → { retriever → format_docs = context, passthrough = question }
                 → prompt_template → LLM → string output
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_RESULTS})

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever


# =============================================================================
# SECTION 6: GLOBAL STATE & LIFECYCLE
# =============================================================================

embeddings = None
vector_store = None
rag_chain = None
retriever = None
llm = None
ingested_sources: List[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load embedding model, vector store, LLM, and build RAG chain."""
    global embeddings, vector_store, rag_chain, retriever, llm

    print("=" * 60)
    print("🌐 Starting RAG Online — Web Link Ingestion Demo")
    print("=" * 60)

    print(f"\n📦 Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    print("   ✅ Embedding model loaded!")

    print(f"\n💾 Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
    vector_store = get_or_create_vector_store(embeddings)
    print("   ✅ Vector store ready!")

    print(f"\n🤖 Loading local LLM via Ollama: {OLLAMA_MODEL_NAME}")
    llm = ChatOllama(model=OLLAMA_MODEL_NAME, temperature=0.1)
    print("   ✅ Local LLM ready!")

    print("\n🔗 Assembling RAG chain...")
    rag_chain, retriever = build_rag_chain(vector_store, llm)
    print("   ✅ RAG chain assembled!")

    print("\n" + "=" * 60)
    print("✨ Ready! Open http://localhost:8002")
    print("=" * 60 + "\n")

    yield
    print("\n👋 Shutting down RAG Online...")


# =============================================================================
# SECTION 7: FASTAPI APP
# =============================================================================

app = FastAPI(
    title="RAG Online — Web Links",
    description="RAG system for web PDF and article ingestion.",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# =============================================================================
# SECTION 8: API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the web UI for link ingestion and chat."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ingest")
async def ingest_link(request: Request):
    """
    Ingests a document from a URL or local file path.

    Supported types:
    • web_pdf     — PDF hosted at a URL
    • web_article — HTML page (Medium, blog, news)
    • local_pdf   — Local PDF file path

    Request body (JSON):
        {"link": "https://medium.com/article..."}
    Or form data:
        link=https://...
    """
    global rag_chain, retriever

    # Accept both JSON and form data
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        link = body.get("link", "").strip()
    else:
        form = await request.form()
        link = form.get("link", "").strip()

    if not link:
        raise HTTPException(status_code=400, detail="Please provide a link.")

    source_name = link

    try:
        print(f"\n🔗 Ingesting from link/path: {source_name}")

        text = extract_text(link)

        if not text or not text.strip():
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract any text from: {source_name}",
            )

        print("  📝 Chunking text...")
        chunks = chunk_text(text, source_name)

        print("  💾 Adding chunks to vector store...")
        vector_store.add_documents(chunks)

        print("  🔗 Rebuilding RAG chain...")
        rag_chain, retriever = build_rag_chain(vector_store, llm)

        source_type = detect_source_type(link)
        ingested_sources.append({
            "source": source_name,
            "chunks": len(chunks),
            "type": source_type,
        })

        print(f"  ✅ Successfully ingested {len(chunks)} chunks from: {source_name}\n")

        return {
            "status": "success",
            "source": source_name,
            "chunks_created": len(chunks),
            "source_type": source_type,
            "message": f"Successfully ingested '{source_name}' ({len(chunks)} chunks)",
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"  ❌ Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/query")
async def query_documents(request: Request):
    """Answers a question using the RAG pipeline."""
    body = await request.json()
    question = body.get("question", "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Please provide a question.")

    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="No documents ingested yet. Ingest a link first.",
        )

    try:
        print(f"\n❓ Query: {question}")

        retrieved_docs = await retriever.ainvoke(question)
        answer = await rag_chain.ainvoke(question)
        sources = list(set(doc.metadata.get("source", "unknown") for doc in retrieved_docs))

        print(f"  💬 Answer: {answer[:100]}...")
        print(f"  📚 Sources: {sources}\n")

        return {"answer": answer, "sources": sources}

    except Exception as e:
        print(f"  ❌ Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Returns ingestion stats."""
    total_chunks = sum(s["chunks"] for s in ingested_sources)
    return {
        "total_documents": len(ingested_sources),
        "total_chunks": total_chunks,
        "sources": ingested_sources,
    }


# =============================================================================
# Run: uvicorn main:app --reload --port 8002
# Open: http://localhost:8002
# =============================================================================
