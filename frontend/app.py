"""
app.py — ScholarSync Streamlit Frontend
========================================
Two-tab interface:
  📚 Upload         — Upload PDF papers to a session
  💬 Chat           — Ask questions about your papers
  ✍️  Writing Assistant — Draft paragraphs, summaries, lit reviews

The frontend calls the FastAPI backend (default: http://localhost:8000).
Set BACKEND_URL in your environment or .env to override.

Run:
    streamlit run frontend/app.py
"""

import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ScholarSync",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; color: #1a1a2e; }
        .sub-header  { font-size: 1rem; color: #555; margin-top: -0.5rem; }
        .source-card {
            background: #f8f9fa;
            border-left: 4px solid #4a90d9;
            padding: 0.6rem 1rem;
            border-radius: 0 6px 6px 0;
            margin: 0.4rem 0;
            font-size: 0.85rem;
        }
        .chunk-text  { color: #333; font-style: italic; }
        .chunk-meta  { color: #666; font-size: 0.78rem; margin-top: 0.2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state initialisation ──────────────────────────────────────────────

def init_state():
    """Initialise Streamlit session state defaults."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "uploaded_papers" not in st.session_state:
        st.session_state.uploaded_papers = []   # list of paper titles
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []      # list of {"role", "content", "sources"}
    if "chunk_count" not in st.session_state:
        st.session_state.chunk_count = 0


init_state()


# ── Helper: API calls ─────────────────────────────────────────────────────────

def api_upload(file_bytes: bytes, filename: str, session_id: str) -> dict:
    """POST /upload — returns the response JSON or raises on error."""
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (filename, file_bytes, "application/pdf")},
        data={"session_id": session_id},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def api_chat(message: str, session_id: str, top_k: int = 5) -> dict:
    """POST /chat — returns answer + sources."""
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={"session_id": session_id, "message": message, "top_k": top_k},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def api_write(prompt: str, session_id: str, write_mode: str, top_k: int = 5) -> dict:
    """POST /write — returns result + sources."""
    response = requests.post(
        f"{BACKEND_URL}/write",
        json={
            "session_id": session_id,
            "prompt": prompt,
            "write_mode": write_mode,
            "top_k": top_k,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def render_sources(sources: list[dict]):
    """Render retrieved source chunks in a collapsible section."""
    if not sources:
        return
    with st.expander(f"📎 Retrieved excerpts ({len(sources)} chunks)", expanded=False):
        for i, src in enumerate(sources, start=1):
            meta = src.get("metadata", {})
            title = meta.get("paper_title", "Unknown")
            page = meta.get("page_number", "?")
            section = meta.get("section", "—")
            text_preview = src.get("text", "")[:300].strip()
            st.markdown(
                f"""
                <div class="source-card">
                    <b>[{i}]</b>
                    <span class="chunk-meta">{title} · p. {page} · {section}</span><br>
                    <span class="chunk-text">"{text_preview}…"</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Session")
    st.code(st.session_state.session_id[:8] + "…", language=None)

    st.markdown(f"**Papers uploaded:** {len(st.session_state.uploaded_papers)}")
    st.markdown(f"**Chunks in store:** {st.session_state.chunk_count}")

    if st.session_state.uploaded_papers:
        st.markdown("**📄 Papers:**")
        for paper in st.session_state.uploaded_papers:
            st.markdown(f"- {paper}")

    st.divider()

    # Allow starting a fresh session
    if st.button("🔄 New Session", use_container_width=True):
        for key in ["session_id", "uploaded_papers", "chat_history", "chunk_count"]:
            st.session_state.pop(key, None)
        init_state()
        st.rerun()

    # Retrieval top_k slider
    st.markdown("### Retrieval settings")
    top_k = st.slider(
        "Chunks to retrieve (top-k)",
        min_value=2,
        max_value=10,
        value=5,
        help="Higher = more context for the LLM, but slower and uses more tokens.",
    )

    st.divider()
    st.caption("ScholarSync v1.0 · Powered by Groq + Llama 3.3 70B")


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">📖 ScholarSync</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">RAG-based academic writing assistant — '
    "chat with your papers & draft citation-backed paragraphs</p>",
    unsafe_allow_html=True,
)
st.divider()


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_upload, tab_chat, tab_write = st.tabs(
    ["📚 Upload Papers", "💬 Chat", "✍️ Writing Assistant"]
)


# ── Tab 1: Upload ─────────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload Research Papers")
    st.caption(
        "Upload one or more PDFs. They'll be parsed, chunked, and embedded "
        "locally — no text is sent to any third party during ingestion."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("⬆️ Ingest Papers", type="primary"):
        progress_bar = st.progress(0, text="Ingesting…")
        errors = []

        for idx, uploaded_file in enumerate(uploaded_files):
            progress_bar.progress(
                (idx) / len(uploaded_files),
                text=f"Ingesting '{uploaded_file.name}'…",
            )
            try:
                result = api_upload(
                    file_bytes=uploaded_file.read(),
                    filename=uploaded_file.name,
                    session_id=st.session_state.session_id,
                )
                st.session_state.uploaded_papers.append(result["title"])
                st.session_state.chunk_count += result["chunks_stored"]
                st.success(
                    f"✅ **{result['title']}** — "
                    f"{result['pages_parsed']} pages, "
                    f"{result['chunks_stored']} chunks stored."
                )
            except requests.exceptions.HTTPError as e:
                detail = ""
                try:
                    detail = e.response.json().get("detail", "")
                except Exception:
                    pass
                errors.append(f"'{uploaded_file.name}': {detail or str(e)}")
            except requests.exceptions.ConnectionError:
                errors.append(
                    f"'{uploaded_file.name}': Cannot reach backend at {BACKEND_URL}. "
                    "Is the FastAPI server running?"
                )

        progress_bar.progress(1.0, text="Done!")

        for err in errors:
            st.error(f"❌ {err}")

        if not errors:
            st.info("Switch to the **Chat** or **Writing Assistant** tab to get started.")

    if not st.session_state.uploaded_papers:
        st.info("👆 Upload at least one PDF to enable chat and writing features.")


# ── Tab 2: Chat ───────────────────────────────────────────────────────────────

with tab_chat:
    st.subheader("Chat with Your Papers")
    st.caption(
        "Ask questions about your uploaded papers. "
        "Every answer is grounded in the actual text with source citations."
    )

    # Render existing conversation
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn["role"] == "assistant" and turn.get("sources"):
                render_sources(turn["sources"])

    # Input
    if not st.session_state.uploaded_papers:
        st.warning("Upload papers in the **Upload Papers** tab first.")
    else:
        user_input = st.chat_input("Ask a question about your papers…")

        if user_input:
            # Show user message immediately
            st.session_state.chat_history.append(
                {"role": "user", "content": user_input, "sources": []}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Call API
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        result = api_chat(
                            message=user_input,
                            session_id=st.session_state.session_id,
                            top_k=top_k,
                        )
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        st.markdown(answer)
                        render_sources(sources)

                        st.session_state.chat_history.append(
                            {"role": "assistant", "content": answer, "sources": sources}
                        )
                    except requests.exceptions.HTTPError as e:
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(f"❌ {detail or str(e)}")
                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"❌ Cannot reach backend at {BACKEND_URL}. "
                            "Is the server running? (`uvicorn main:app --reload`)"
                        )

    # Clear chat button
    if st.session_state.chat_history:
        if st.button("🗑️ Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


# ── Tab 3: Writing Assistant ──────────────────────────────────────────────────

with tab_write:
    st.subheader("✍️ Writing Assistant")
    st.caption(
        "Generate citation-backed academic paragraphs from your uploaded papers. "
        "Choose a mode, describe your topic, and ScholarSync will draft it for you."
    )

    if not st.session_state.uploaded_papers:
        st.warning("Upload papers in the **Upload Papers** tab first.")
    else:
        write_mode_options = {
            "📋 Summarize": "summarize",
            "⚖️ Compare": "compare",
            "📖 Literature Review": "literature_review",
            "🔖 Suggest Citations": "citations",
        }

        col1, col2 = st.columns([1, 2])

        with col1:
            mode_label = st.radio(
                "Writing mode",
                options=list(write_mode_options.keys()),
                index=2,  # default: Literature Review
            )
            write_mode = write_mode_options[mode_label]

            mode_descriptions = {
                "summarize": "Summarize the key findings and contributions from your papers.",
                "compare": "Compare methodologies, findings, or arguments across papers.",
                "literature_review": "Draft a cohesive literature review paragraph.",
                "citations": "Identify citable claims and show how to use them.",
            }
            st.info(mode_descriptions[write_mode])

        with col2:
            topic_prompt = st.text_area(
                "Describe your topic or provide additional context",
                placeholder=(
                    "e.g. 'Focus on transformer-based models for NLP tasks' or "
                    "'Compare the datasets used across the papers'"
                ),
                height=120,
            )

            generate_btn = st.button("🚀 Generate", type="primary")

        if generate_btn:
            if not topic_prompt.strip():
                st.warning("Please enter a topic or prompt above.")
            else:
                with st.spinner("Drafting your content…"):
                    try:
                        result = api_write(
                            prompt=topic_prompt,
                            session_id=st.session_state.session_id,
                            write_mode=write_mode,
                            top_k=top_k,
                        )
                        st.divider()
                        st.markdown("### Generated Content")
                        st.markdown(result["result"])
                        render_sources(result.get("sources", []))

                        # Copy-to-clipboard via a text area
                        st.divider()
                        st.text_area(
                            "Copy output",
                            value=result["result"],
                            height=200,
                            label_visibility="collapsed",
                        )

                    except requests.exceptions.HTTPError as e:
                        detail = ""
                        try:
                            detail = e.response.json().get("detail", "")
                        except Exception:
                            pass
                        st.error(f"❌ {detail or str(e)}")
                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"❌ Cannot reach backend at {BACKEND_URL}. "
                            "Is the server running?"
                        )
