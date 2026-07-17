"""
retrieval.py — Semantic Retrieval Layer
========================================
Given a natural-language query and a session_id, this module:
  1. Embeds the query using the same model used at ingest time
  2. Queries ChromaDB for the top-k most relevant chunks
  3. Returns structured results ready for the generation layer

Design notes:
  - Retrieval is session-scoped, so each user's papers are kept isolated.
  - top_k defaults to 5 (good balance of context vs. prompt length).
  - Results include full metadata so the LLM can cite paper + page.
"""

import logging
from typing import Optional

from vectorstore import VectorStore

logger = logging.getLogger(__name__)

# Default number of chunks to retrieve; callers may override
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    session_id: str,
    top_k: int = DEFAULT_TOP_K,
    vectorstore: Optional[VectorStore] = None,
) -> list[dict]:
    """
    Retrieve the top-k most semantically relevant chunks for `query`.

    Args:
        query:       The user's question or writing task description.
        session_id:  Identifies which ChromaDB collection to search.
        top_k:       How many chunks to return (typically 4-6).
        vectorstore: Optional pre-built VectorStore (useful for unit tests).

    Returns:
        List of chunk dicts sorted by relevance (most relevant first):
            {
                "text": str,
                "metadata": {
                    "paper_title": str,
                    "page_number": int,
                    "section": str,
                    "source_file": str,
                },
                "distance": float,   # cosine distance; lower = more similar
                "id": str,
            }

    Raises:
        ValueError: if query is empty or collection has no documents.
    """
    if not query.strip():
        raise ValueError("Query must not be empty.")

    vs = vectorstore or VectorStore(session_id=session_id)

    if vs.count() == 0:
        raise ValueError(
            f"No documents found for session '{session_id}'. "
            "Please upload at least one PDF before querying."
        )

    # Clamp top_k to the number of available chunks
    effective_k = min(top_k, vs.count())
    chunks = vs.query(query_text=query, top_k=effective_k)

    logger.info(
        f"Retrieved {len(chunks)} chunks for session='{session_id}' | "
        f"query='{query[:60]}…'"
    )
    return chunks


def format_context_block(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a numbered context block for the LLM prompt.

    Example output:
        [1] "Transformers use self-attention…"
            Source: Attention Is All You Need, p. 3 (Section: Methods)

        [2] …

    This format makes it easy for the LLM to produce inline citations like
    "[1]" or "(Vaswani et al., p. 3)" in its response.
    """
    if not chunks:
        return "No relevant excerpts found."

    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        title = meta.get("paper_title", "Unknown")
        page = meta.get("page_number", "?")
        section = meta.get("section", "Unknown")

        lines.append(f'[{i}] "{chunk["text"].strip()}"')
        lines.append(f"    Source: {title}, p. {page} (Section: {section})\n")

    return "\n".join(lines)


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test retrieval pipeline")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument(
        "--session-id",
        default="test_session",
        help="Session/collection ID (default: test_session)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to retrieve (default: {DEFAULT_TOP_K})",
    )
    args = parser.parse_args()

    try:
        chunks = retrieve(args.query, session_id=args.session_id, top_k=args.top_k)
        print(f"\n🔍 Top {len(chunks)} results for: '{args.query}'\n")
        print(format_context_block(chunks))
    except ValueError as e:
        print(f"\n❌ Error: {e}")
