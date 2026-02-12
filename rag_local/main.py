"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              RAG Local — PDF File Upload Demo                              ║
║                     Workshop Edition v2.0                                  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  This project handles LOCAL PDF file ingestion only.                       ║
║                                                                            ║
║  Pipeline:                                                                 ║
║  1. UPLOAD  → User uploads a PDF file via the web UI                       ║
║  2. EXTRACT → Extract text from PDF bytes using pypdf                      ║
║  3. CHUNK   → Split text into overlapping chunks                           ║
║  4. EMBED   → Convert chunks into vector embeddings                        ║
║  5. STORE   → Persist embeddings in ChromaDB                               ║
║  6. QUERY   → Retrieve relevant chunks and generate answers via LLM        ║
║                                                                            ║
║  Run:  uvicorn main:app --reload --port 8001                               ║
║  Open: http://localhost:8001                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import io                          # In-memory binary streams (for PDF bytes)
from typing import List            # Type hints
from contextlib import asynccontextmanager

import pypdf                       # PDF text extraction

from fastapi import FastAPI, UploadFile, File, HTTPException
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

CHROMA_PERSIST_DIR = "./chroma_db_local"          # Separate DB for local project
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
# SECTION 4: TEXT EXTRACTION — PDF BYTES
# =============================================================================

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts text from raw PDF bytes.

    How it works:
    1. Wrap the bytes in an in-memory file-like object (BytesIO)
    2. Use pypdf to parse the PDF structure
    3. Iterate through each page and extract its text content
    4. Concatenate all page texts into a single string

    Args:
        pdf_bytes: Raw bytes of a PDF file.

    Returns:
        The extracted text as a single string.
    """
    pdf_file = io.BytesIO(pdf_bytes)
    reader = pypdf.PdfReader(pdf_file)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    return text


# =============================================================================
# SECTION 5: VECTOR STORE & RAG CHAIN
# =============================================================================

def chunk_text(text: str, source_name: str) -> List[Document]:
    """
    Splits text into smaller, overlapping chunks for retrieval.

    The RecursiveCharacterTextSplitter tries to split on natural boundaries:
    paragraphs → sentences → words → characters (in that order).
    """
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
    Assembles the RAG chain using LCEL (LangChain Expression Language).

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
    print("📄 Starting RAG Local — PDF Upload Demo")
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
    print("✨ Ready! Open http://localhost:8001")
    print("=" * 60 + "\n")

    yield
    print("\n👋 Shutting down RAG Local...")


# =============================================================================
# SECTION 7: FASTAPI APP
# =============================================================================

app = FastAPI(
    title="RAG Local — PDF Upload",
    description="RAG system for local PDF file uploads.",
    version="2.0.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


# =============================================================================
# SECTION 8: API ENDPOINTS
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    """Serves the web UI for PDF upload and chat."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """
    Ingests an uploaded PDF file into the vector store.

    Pipeline:
    1. Read uploaded file bytes
    2. Extract text from PDF
    3. Chunk the text
    4. Embed and store in ChromaDB
    5. Rebuild RAG chain
    """
    global rag_chain, retriever

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    source_name = file.filename

    try:
        print(f"\n📥 Ingesting uploaded file: {source_name}")

        file_bytes = await file.read()
        text = extract_text_from_pdf_bytes(file_bytes)

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

        ingested_sources.append({
            "source": source_name,
            "chunks": len(chunks),
            "type": "uploaded_pdf",
        })

        print(f"  ✅ Successfully ingested {len(chunks)} chunks from: {source_name}\n")

        return {
            "status": "success",
            "source": source_name,
            "chunks_created": len(chunks),
            "message": f"Successfully ingested '{source_name}' ({len(chunks)} chunks)",
        }

    except HTTPException:
        raise
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
            detail="No documents ingested yet. Upload a PDF first.",
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
# Run: uvicorn main:app --reload --port 8001
# Open: http://localhost:8001
# =============================================================================
