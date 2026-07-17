"""
ingest.py — PDF Parsing, Chunking, and Embedding Pipeline
==========================================================
Handles the full ingestion lifecycle for a single PDF:
  1. Parse text + metadata with PyMuPDF
  2. Detect section headings with a simple heuristic
  3. Recursively split text into ~400-word chunks with ~50-word overlap
  4. Generate embeddings with sentence-transformers (all-MiniLM-L6-v2)
  5. Store chunks + embeddings in ChromaDB via the vectorstore wrapper

Usage (CLI test):
    python ingest.py path/to/paper.pdf --session-id test_session
"""

import re
import uuid
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from vectorstore import VectorStore

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CHUNK_SIZE_WORDS = 400      # target chunk size in words
CHUNK_OVERLAP_WORDS = 50    # overlap between consecutive chunks in words

# Patterns that look like section headings (all-caps or numbered, short lines)
_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[\s.]+[A-Z]"  # numbered: "1. Introduction", "2.3 Method"
    r"|[A-Z][A-Z\s]{3,40}$"      # all-caps short line: "INTRODUCTION"
    r")",
    re.MULTILINE,
)


# ── PDF Parsing ───────────────────────────────────────────────────────────────

def parse_pdf(pdf_path: str) -> list[dict]:
    """
    Open a PDF and extract per-page text with metadata.

    Returns a list of page dicts:
        {
            "page_number": int,      # 1-indexed
            "text": str,             # raw page text
            "title": str,            # inferred from first-page or filename
        }

    Raises ValueError for empty or unreadable PDFs.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(path))

    if doc.page_count == 0:
        raise ValueError(f"PDF has no pages: {pdf_path}")

    # Try to get a human-readable title from PDF metadata; fall back to filename
    meta = doc.metadata or {}
    title = (meta.get("title") or "").strip() or path.stem

    pages = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text("text")  # plain text extraction
        if text.strip():              # skip blank pages
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "title": title,
            })

    doc.close()

    if not pages:
        raise ValueError(
            f"No extractable text found in '{path.name}'. "
            "The PDF may be scanned/image-based (OCR not supported in v1)."
        )

    logger.info(f"Parsed '{title}': {len(pages)} pages with text")
    return pages


# ── Section Detection ─────────────────────────────────────────────────────────

def detect_section(text_before: str) -> Optional[str]:
    """
    Heuristically identify the last section heading that appeared before a
    given position in the text.  Returns the heading string or None.
    """
    matches = list(_HEADING_RE.finditer(text_before))
    if matches:
        return matches[-1].group(0).strip()
    return None


# ── Recursive Text Splitter ───────────────────────────────────────────────────

def _split_words(text: str) -> list[str]:
    """Split text into a list of whitespace-delimited tokens."""
    return text.split()


def recursive_chunk(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """
    Recursively split `text` into chunks of ~`chunk_size` words with
    `overlap`-word overlap between consecutive chunks.

    Strategy (greedy with separator hierarchy):
      - First try to split on paragraph breaks (double newline)
      - Then on single newlines
      - Then on sentence-ending punctuation
      - Finally hard-split by word count

    This mimics LangChain's RecursiveCharacterTextSplitter but operates
    on word counts rather than character counts.
    """
    words = _split_words(text)
    if len(words) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        # Advance by chunk_size minus overlap so next chunk re-uses the tail
        start += chunk_size - overlap

    return chunks


# ── Full Ingestion Pipeline ────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: str,
    session_id: str,
    vectorstore: Optional[VectorStore] = None,
) -> dict:
    """
    Full pipeline: parse → chunk → embed → store.

    Args:
        pdf_path:    Local path to the uploaded PDF.
        session_id:  User/session identifier; used as ChromaDB collection name.
        vectorstore: Optional pre-built VectorStore; created if not provided.

    Returns:
        {
            "title": str,
            "pages_parsed": int,
            "chunks_stored": int,
            "session_id": str,
        }
    """
    # 1 — Parse
    pages = parse_pdf(pdf_path)
    paper_title = pages[0]["title"]

    # 2 — Build VectorStore for this session (lazy-creates collection)
    if vectorstore is None:
        vectorstore = VectorStore(session_id=session_id)

    # 3 — Chunk each page, preserving metadata
    all_texts: list[str] = []
    all_metadatas: list[dict] = []
    all_ids: list[str] = []

    for page_info in pages:
        page_text = page_info["text"]
        page_num = page_info["page_number"]

        chunks = recursive_chunk(page_text)

        for chunk_text in chunks:
            if not chunk_text.strip():
                continue

            # Best-effort section heading detection
            # Look for a heading in the text before this chunk on the page
            position = page_text.find(chunk_text[:50])  # anchor on first 50 chars
            preceding = page_text[:position] if position > 0 else ""
            section = detect_section(preceding) or "Unknown"

            chunk_id = str(uuid.uuid4())
            all_texts.append(chunk_text)
            all_metadatas.append({
                "paper_title": paper_title,
                "page_number": page_num,
                "section": section,
                "source_file": Path(pdf_path).name,
            })
            all_ids.append(chunk_id)

    if not all_texts:
        raise ValueError(f"No text chunks produced from '{paper_title}'.")

    # 4 — Embed + store (vectorstore handles batching)
    vectorstore.add(
        texts=all_texts,
        metadatas=all_metadatas,
        ids=all_ids,
    )

    logger.info(
        f"Ingested '{paper_title}': {len(pages)} pages → {len(all_texts)} chunks"
    )
    return {
        "title": paper_title,
        "pages_parsed": len(pages),
        "chunks_stored": len(all_texts),
        "session_id": session_id,
    }


# ── CLI Test Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test PDF ingestion pipeline")
    parser.add_argument("pdf_path", help="Path to a PDF file")
    parser.add_argument(
        "--session-id",
        default="test_session",
        help="Session/collection ID for ChromaDB (default: test_session)",
    )
    args = parser.parse_args()

    try:
        result = ingest_pdf(args.pdf_path, session_id=args.session_id)
        print("\n✅ Ingestion complete:")
        for k, v in result.items():
            print(f"   {k}: {v}")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
