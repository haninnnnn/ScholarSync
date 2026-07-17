"""
generation.py — LLM Generation with Groq API
=============================================
Two generation modes powered by Llama 3.3 70B via Groq:

  "chat"  — Grounded Q&A: answer the user's question using ONLY the
             retrieved excerpts, citing paper + page for each claim.

  "write" — Academic writing assistant: synthesize multiple sources
             into a coherent paragraph with inline citations.
             Sub-modes: summarize | compare | literature_review | citations

Environment variable required:
    GROQ_API_KEY  — your Groq API key (never hardcoded here)

Usage (CLI test):
    python generation.py --mode chat --session-id test_session "What is BERT?"
"""

import logging
import os
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

from retrieval import retrieve, format_context_block

load_dotenv()  # load .env if present (development convenience)

logger = logging.getLogger(__name__)

# ── Groq client (lazy init so the module can be imported without a key) ────────
_groq_client = None

def get_groq_client() -> Groq:
    """Return the shared Groq client, raising a clear error if key is missing."""
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Add it to your .env file or export it in your shell."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Model config ──────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 1024
TEMPERATURE = 0.3   # lower = more factual/consistent for academic use


# ── Prompt templates ──────────────────────────────────────────────────────────

_CHAT_SYSTEM_PROMPT = """\
You are ScholarSync, an academic research assistant. Your job is to answer \
the user's question using ONLY the excerpts provided below. 

Rules:
1. Base every claim on the provided excerpts. Do not use outside knowledge.
2. After each claim, cite the source using the format [N] where N is the \
excerpt number.
3. If the excerpts do not contain enough information to answer, say so clearly \
rather than guessing.
4. Be concise and precise. Prefer bullet points for multi-part answers.
5. If quoting directly, use quotation marks and cite immediately after.
"""

_WRITE_SYSTEM_PROMPT = """\
You are ScholarSync, an academic writing assistant. Your job is to help \
students write high-quality academic content grounded in their uploaded sources.

Rules:
1. Write in formal academic English (third person, passive voice where \
appropriate, no contractions).
2. Every factual claim MUST be supported by one of the provided excerpts and \
cited inline as [N].
3. Synthesize information from multiple sources where possible — do not just \
paraphrase one excerpt.
4. Do not fabricate facts, statistics, or citations.
5. End with a "References" line listing the cited sources in the format:
   [N] Author/Title, page P.
"""

_WRITE_MODE_INSTRUCTIONS = {
    "summarize": (
        "Write a concise academic summary (2-3 paragraphs) of the key findings "
        "and contributions described in the provided excerpts."
    ),
    "compare": (
        "Write a comparative analysis paragraph discussing similarities and "
        "differences between the approaches, findings, or methods described "
        "in the provided excerpts. Highlight agreements and contradictions."
    ),
    "literature_review": (
        "Draft a literature review paragraph that synthesizes the provided "
        "excerpts into a coherent narrative. Group related ideas, identify "
        "themes, and show how the sources relate to each other."
    ),
    "citations": (
        "Identify the key claims in the provided excerpts and suggest how "
        "each could be used as a citation in an academic paper. For each, "
        "provide: the claim, the suggested citation context, and the exact "
        "excerpt number."
    ),
}


# ── Core generation function ──────────────────────────────────────────────────

def generate(
    query: str,
    session_id: str,
    mode: Literal["chat", "write"] = "chat",
    write_mode: Literal["summarize", "compare", "literature_review", "citations"] = "literature_review",
    conversation_history: list[dict] | None = None,
    top_k: int = 5,
) -> dict:
    """
    Retrieve relevant chunks and generate a grounded LLM response.

    Args:
        query:                The user's question or writing prompt.
        session_id:           ChromaDB collection to search.
        mode:                 "chat" for Q&A, "write" for academic writing.
        write_mode:           Sub-mode used when mode="write".
        conversation_history: Prior messages as [{"role": "...", "content": "..."}].
                              Enables follow-up questions in chat mode.
        top_k:                Number of chunks to retrieve (4-6 recommended).

    Returns:
        {
            "answer": str,          # LLM response
            "sources": list[dict],  # retrieved chunks with metadata
            "context_block": str,   # formatted excerpts shown to LLM
        }
    """
    # 1 — Retrieve relevant chunks
    chunks = retrieve(query, session_id=session_id, top_k=top_k)
    context_block = format_context_block(chunks)

    # 2 — Build system + user messages
    if mode == "chat":
        system_prompt = _CHAT_SYSTEM_PROMPT
        user_content = (
            f"Excerpts from uploaded papers:\n\n"
            f"{context_block}\n\n"
            f"---\n"
            f"Question: {query}"
        )
    else:  # write
        instruction = _WRITE_MODE_INSTRUCTIONS.get(
            write_mode, _WRITE_MODE_INSTRUCTIONS["literature_review"]
        )
        system_prompt = _WRITE_SYSTEM_PROMPT
        user_content = (
            f"Task: {instruction}\n\n"
            f"Excerpts from uploaded papers:\n\n"
            f"{context_block}\n\n"
            f"---\n"
            f"Additional context from the student: {query}"
        )

    # 3 — Build the messages list (include history for chat mode)
    messages = [{"role": "system", "content": system_prompt}]

    if mode == "chat" and conversation_history:
        # Append prior turns (skip the system message if already present)
        for turn in conversation_history:
            if turn.get("role") in ("user", "assistant"):
                messages.append(turn)

    messages.append({"role": "user", "content": user_content})

    # 4 — Call Groq
    client = get_groq_client()
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )
    except Exception as e:
        logger.error(f"Groq API call failed: {e}")
        raise RuntimeError(f"LLM generation failed: {e}") from e

    answer = response.choices[0].message.content.strip()
    logger.info(
        f"Generated response | mode={mode} | session='{session_id}' | "
        f"tokens_used={response.usage.total_tokens}"
    )

    return {
        "answer": answer,
        "sources": chunks,
        "context_block": context_block,
    }


# ── CLI test ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test generation pipeline")
    parser.add_argument("query", help="Question or writing prompt")
    parser.add_argument(
        "--session-id",
        default="test_session",
        help="Session/collection ID (default: test_session)",
    )
    parser.add_argument(
        "--mode",
        choices=["chat", "write"],
        default="chat",
        help="Generation mode (default: chat)",
    )
    parser.add_argument(
        "--write-mode",
        choices=list(_WRITE_MODE_INSTRUCTIONS.keys()),
        default="literature_review",
        help="Sub-mode for write (default: literature_review)",
    )
    args = parser.parse_args()

    try:
        result = generate(
            query=args.query,
            session_id=args.session_id,
            mode=args.mode,
            write_mode=args.write_mode,
        )
        print(f"\n{'='*60}")
        print(f"Mode: {args.mode}" + (f" / {args.write_mode}" if args.mode == "write" else ""))
        print(f"{'='*60}\n")
        print(result["answer"])
        print(f"\n{'─'*60}")
        print("Retrieved excerpts:")
        print(result["context_block"])
    except (ValueError, EnvironmentError, RuntimeError) as e:
        print(f"\n❌ Error: {e}")
