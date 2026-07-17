"""
main.py — ScholarSync FastAPI Backend
======================================
Endpoints:
    POST /upload          — Parse, chunk, embed, and store a PDF
    POST /chat            — Grounded Q&A over uploaded papers
    POST /write           — Academic writing assistant (synthesis/draft)
    GET  /session/{id}    — Info about a session (chunk count, etc.)
    DELETE /session/{id}  — Clear a session's vector store

Run locally:
    uvicorn main:app --reload --port 8000

Environment variables (see .env.example):
    GROQ_API_KEY  — required for /chat and /write endpoints
"""

import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from generation import generate
from ingest import ingest_pdf
from vectorstore import VectorStore

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent
_UPLOADS_DIR = _ROOT / "data" / "uploads"
_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="ScholarSync API",
    description="RAG-based academic writing assistant — upload papers, chat, and write.",
    version="1.0.0",
)

# Allow requests from the Streamlit frontend (all origins for dev; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory conversation history ────────────────────────────────────────────
# Maps session_id → list of {"role": "user"|"assistant", "content": str}
# NOTE: This is reset when the server restarts. For production, use Redis/DB.
_conversation_histories: dict[str, list[dict]] = {}


# ── Pydantic request/response models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[dict]


class WriteRequest(BaseModel):
    session_id: str
    prompt: str
    write_mode: Literal["summarize", "compare", "literature_review", "citations"] = "literature_review"
    top_k: int = 5


class WriteResponse(BaseModel):
    session_id: str
    result: str
    sources: list[dict]


class UploadResponse(BaseModel):
    session_id: str
    title: str
    pages_parsed: int
    chunks_stored: int
    message: str


class SessionInfo(BaseModel):
    session_id: str
    chunk_count: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_upload(upload_file: UploadFile, session_id: str) -> Path:
    """
    Save an uploaded file to disk under data/uploads/<session_id>/.
    Returns the saved file path.
    """
    session_dir = _UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    # Preserve original filename; sanitise to avoid path traversal
    safe_name = Path(upload_file.filename or "upload.pdf").name
    dest = session_dir / safe_name

    with dest.open("wb") as f:
        shutil.copyfileobj(upload_file.file, f)

    return dest


def _check_groq_key() -> None:
    """Raise a clear 503 if the Groq API key is not configured."""
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "GROQ_API_KEY is not configured on the server. "
                "Set it in your .env file and restart the server."
            ),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def root():
    """Health check — confirms the API is running."""
    return {"status": "ok", "service": "ScholarSync API v1.0.0"}


@app.post("/upload", response_model=UploadResponse, tags=["Ingest"])
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    session_id: Optional[str] = Form(
        default=None,
        description="Session ID (auto-generated if omitted)",
    ),
):
    """
    Upload and ingest a PDF paper.

    - Parses text with PyMuPDF
    - Chunks into ~400-word segments with 50-word overlap
    - Embeds with all-MiniLM-L6-v2 (runs locally)
    - Stores in ChromaDB under the given session_id

    Multiple papers can be uploaded to the same session_id; retrieval
    searches across all of them.
    """
    # Validate file type
    if not (upload_file_name := (file.filename or "")).lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Auto-generate a session_id if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    # Save to disk
    try:
        saved_path = _save_upload(file, session_id)
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # Ingest
    try:
        result = ingest_pdf(str(saved_path), session_id=session_id)
    except ValueError as e:
        # Empty / image-only PDF
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")

    return UploadResponse(
        session_id=session_id,
        title=result["title"],
        pages_parsed=result["pages_parsed"],
        chunks_stored=result["chunks_stored"],
        message=(
            f"Successfully ingested '{result['title']}' "
            f"({result['chunks_stored']} chunks stored)."
        ),
    )


@app.post("/chat", response_model=ChatResponse, tags=["Generation"])
def chat(request: ChatRequest):
    """
    Chat with your uploaded papers.

    Retrieves the most relevant excerpts for the user's message and asks
    Llama 3.3 70B to answer using only those excerpts, with citations.

    Conversation history is maintained server-side per session_id so
    follow-up questions work correctly.
    """
    _check_groq_key()

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message must not be empty.")

    # Retrieve or initialise conversation history for this session
    history = _conversation_histories.setdefault(request.session_id, [])

    try:
        result = generate(
            query=request.message,
            session_id=request.session_id,
            mode="chat",
            conversation_history=history,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    # Persist this turn in history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": result["answer"]})

    # Keep history bounded to last 20 turns to avoid prompt bloat
    if len(history) > 20:
        _conversation_histories[request.session_id] = history[-20:]

    return ChatResponse(
        session_id=request.session_id,
        answer=result["answer"],
        sources=result["sources"],
    )


@app.post("/write", response_model=WriteResponse, tags=["Generation"])
def write(request: WriteRequest):
    """
    Academic writing assistant.

    Modes:
    - **summarize**         — summarize key findings from uploaded papers
    - **compare**           — compare approaches / findings across papers
    - **literature_review** — draft a literature review paragraph
    - **citations**         — suggest how to cite specific claims

    Each response is grounded in retrieved excerpts with inline citations.
    """
    _check_groq_key()

    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty.")

    try:
        result = generate(
            query=request.prompt,
            session_id=request.session_id,
            mode="write",
            write_mode=request.write_mode,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except EnvironmentError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return WriteResponse(
        session_id=request.session_id,
        result=result["answer"],
        sources=result["sources"],
    )


@app.get("/session/{session_id}", response_model=SessionInfo, tags=["Session"])
def session_info(session_id: str):
    """Return the chunk count and status for a given session."""
    try:
        vs = VectorStore(session_id=session_id)
        count = vs.count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SessionInfo(session_id=session_id, chunk_count=count)


@app.delete("/session/{session_id}", tags=["Session"])
def delete_session(session_id: str):
    """
    Delete a session's vector store and conversation history.
    Also removes uploaded PDFs from disk.
    """
    # Clear ChromaDB collection
    try:
        vs = VectorStore(session_id=session_id)
        vs.delete_collection()
    except Exception as e:
        logger.warning(f"Could not delete vector collection: {e}")

    # Clear conversation history
    _conversation_histories.pop(session_id, None)

    # Remove uploaded PDFs
    session_upload_dir = _UPLOADS_DIR / session_id
    if session_upload_dir.exists():
        shutil.rmtree(session_upload_dir)

    return {"message": f"Session '{session_id}' deleted successfully."}
