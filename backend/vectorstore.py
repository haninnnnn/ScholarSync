"""
vectorstore.py — ChromaDB Wrapper
==================================
Thin abstraction over ChromaDB that:
  - Persists collections to disk under data/chroma/
  - Generates embeddings locally with sentence-transformers
  - Exposes add() and query() methods used by ingest.py and retrieval.py
  - Supports one collection per session_id (isolates papers by user session)

The embedding model (all-MiniLM-L6-v2) is loaded once at module import time
and reused across all VectorStore instances to avoid redundant model loads.
"""

import logging
import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Logging ───────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Resolve relative to this file so the backend can be run from any cwd
_BACKEND_DIR = Path(__file__).parent
_CHROMA_DIR = _BACKEND_DIR.parent / "data" / "chroma"

# ── Embedding model (singleton) ───────────────────────────────────────────────
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Lazily instantiated so tests can import the module without triggering download
_embed_model: Optional[SentenceTransformer] = None


def get_embed_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first call."""
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading embedding model '{_EMBED_MODEL_NAME}' …")
        _embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
        logger.info("Embedding model loaded.")
    return _embed_model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings and return a list of float vectors.
    Uses the singleton model to avoid reloads.
    """
    model = get_embed_model()
    # encode() returns a numpy array; convert each row to a plain Python list
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


# ── VectorStore class ─────────────────────────────────────────────────────────

class VectorStore:
    """
    Per-session ChromaDB collection wrapper.

    Each session_id maps to one Chroma collection stored under
    data/chroma/<session_id>/. This cleanly separates different users'
    uploaded papers.

    Example usage:
        vs = VectorStore(session_id="alice_session")
        vs.add(texts=["chunk 1", "chunk 2"], metadatas=[{...}, {...}], ids=["id1", "id2"])
        results = vs.query("What is the methodology?", top_k=5)
    """

    def __init__(self, session_id: str, chroma_dir: Optional[str] = None):
        """
        Args:
            session_id:  Identifies the collection. Should be URL-safe
                         (alphanumeric + underscores/hyphens).
            chroma_dir:  Override the default persistence directory.
        """
        self.session_id = session_id

        persist_path = str(chroma_dir or _CHROMA_DIR)
        os.makedirs(persist_path, exist_ok=True)

        # PersistentClient writes to disk automatically on every mutation
        self._client = chromadb.PersistentClient(
            path=persist_path,
            settings=Settings(anonymized_telemetry=False),
        )

        # get_or_create so repeated uploads into the same session accumulate
        self._collection = self._client.get_or_create_collection(
            name=self._sanitize_name(session_id),
            metadata={"hnsw:space": "cosine"},  # cosine similarity
        )

        logger.info(
            f"VectorStore ready | session='{session_id}' | "
            f"existing chunks={self._collection.count()}"
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """
        ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens,
        start/end with alphanumeric.  This sanitizer makes any string safe.
        """
        import re
        safe = re.sub(r"[^a-zA-Z0-9\-]", "-", name)
        safe = re.sub(r"-{2,}", "-", safe).strip("-")
        safe = safe[:63]
        if len(safe) < 3:
            safe = safe.ljust(3, "0")
        return safe

    # ── Public API ────────────────────────────────────────────────────────────

    def add(
        self,
        texts: list[str],
        metadatas: list[dict],
        ids: list[str],
        batch_size: int = 64,
    ) -> None:
        """
        Embed `texts` and upsert them into the collection.

        Args:
            texts:      Raw text strings (one per chunk).
            metadatas:  Parallel list of metadata dicts (paper_title, page, …).
            ids:        Parallel list of unique string IDs.
            batch_size: Number of chunks embedded + stored per round-trip
                        (avoids memory spikes on large papers).
        """
        if not texts:
            logger.warning("add() called with empty texts list — nothing stored.")
            return

        total = len(texts)
        stored = 0

        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_texts = texts[start:end]
            batch_meta = metadatas[start:end]
            batch_ids = ids[start:end]

            embeddings = embed_texts(batch_texts)

            self._collection.upsert(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=batch_meta,
                ids=batch_ids,
            )
            stored += len(batch_texts)
            logger.debug(f"Stored batch {start}–{end} ({stored}/{total})")

        logger.info(f"Stored {stored} chunks in collection '{self.session_id}'")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Embed `query_text` and return the top-k most similar chunks.

        Returns a list of result dicts, sorted by similarity (best first):
            {
                "text": str,
                "metadata": dict,   # paper_title, page_number, section, …
                "distance": float,  # cosine distance (lower = more similar)
                "id": str,
            }
        """
        if not query_text.strip():
            raise ValueError("Query text must not be empty.")

        query_embedding = embed_texts([query_text])[0]

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        # Unpack the nested lists that ChromaDB returns for batch queries
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]
        ids = results["ids"][0]

        return [
            {
                "text": doc,
                "metadata": meta,
                "distance": dist,
                "id": chunk_id,
            }
            for doc, meta, dist, chunk_id in zip(docs, metas, dists, ids)
        ]

    def count(self) -> int:
        """Return the total number of stored chunks in this session."""
        return self._collection.count()

    def delete_collection(self) -> None:
        """Remove the entire collection from ChromaDB (use with caution)."""
        self._client.delete_collection(self._sanitize_name(self.session_id))
        logger.info(f"Deleted collection '{self.session_id}'")


# ── CLI verification script ───────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== VectorStore smoke test ===\n")

    vs = VectorStore(session_id="smoke_test")

    sample_texts = [
        "Transformer models use self-attention to capture long-range dependencies.",
        "BERT is pre-trained on masked language modeling and next sentence prediction.",
        "GPT models are auto-regressive and trained on next-token prediction.",
        "Attention mechanisms allow the model to focus on relevant parts of the input.",
    ]
    sample_meta = [
        {"paper_title": "Test Paper", "page_number": i + 1, "section": "Background", "source_file": "test.pdf"}
        for i in range(len(sample_texts))
    ]
    sample_ids = [f"test-{i}" for i in range(len(sample_texts))]

    vs.add(texts=sample_texts, metadatas=sample_meta, ids=sample_ids)
    print(f"Stored {vs.count()} chunks\n")

    query = "How does self-attention work?"
    print(f"Query: '{query}'\nTop-2 results:\n")
    for r in vs.query(query, top_k=2):
        print(f"  [{r['distance']:.4f}] ({r['metadata']['paper_title']}, p.{r['metadata']['page_number']})")
        print(f"  {r['text'][:120]}\n")

    # Clean up smoke-test collection
    vs.delete_collection()
    print("✅ Smoke test passed — collection cleaned up.")
