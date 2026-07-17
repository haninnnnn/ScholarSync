# 📖 ScholarSync

**A RAG-based academic writing assistant** — upload research papers, chat with them, and draft citation-backed paragraphs grounded in your sources.

Built as a student portfolio project demonstrating Retrieval-Augmented Generation (RAG) end-to-end: PDF ingestion → local embeddings → vector search → LLM generation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **PDF Ingestion** | Upload multiple papers; parsed with PyMuPDF, chunked recursively, embedded locally |
| 💬 **Grounded Chat** | Q&A over your papers — every answer cites the source paper and page number |
| ✍️ **Writing Assistant** | Four modes: Summarize, Compare, Draft Literature Review, Suggest Citations |
| 🔒 **Local Embeddings** | `all-MiniLM-L6-v2` runs on your machine — no text sent to third parties during ingest |
| 🗄️ **Persistent VectorDB** | ChromaDB persisted to disk; survives server restarts |
| 🌐 **Streamlit UI** | Clean two-tab interface; no frontend framework knowledge needed |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit Frontend                        │
│   Upload Tab │ Chat Tab │ Writing Assistant Tab             │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP (REST)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend                            │
│   POST /upload   POST /chat   POST /write                   │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  ingest.py          retrieval.py    generation.py    │  │
│  │  ┌────────────┐    ┌─────────────┐  ┌─────────────┐  │  │
│  │  │ PyMuPDF    │    │ Embed query │  │ Build prompt│  │  │
│  │  │ parse PDF  │    │ ChromaDB    │  │ Groq API    │  │  │
│  │  │ chunk text │    │ top-k search│  │ Llama 3.3   │  │  │
│  │  └────┬───────┘    └──────┬──────┘  └─────────────┘  │  │
│  │       │                   │                           │  │
│  │  ┌────▼───────────────────▼──────────────────────┐   │  │
│  │  │            vectorstore.py                      │   │  │
│  │  │   sentence-transformers (all-MiniLM-L6-v2)    │   │  │
│  │  │   ChromaDB (persisted to data/chroma/)        │   │  │
│  │  └───────────────────────────────────────────────┘   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

**Upload:**
```
PDF file → PyMuPDF (parse pages) → recursive_chunk (~400 words, 50 overlap)
        → sentence-transformers (embed locally) → ChromaDB (persist to disk)
```

**Chat / Write:**
```
User query → sentence-transformers (embed) → ChromaDB (top-k cosine search)
           → retrieved chunks → prompt template → Groq API (Llama 3.3 70B)
           → grounded answer with inline citations [1], [2], …
```

---

## 📁 Project Structure

```
scholarsync/
├── backend/
│   ├── main.py          # FastAPI app: /upload, /chat, /write endpoints
│   ├── ingest.py        # PDF parsing, recursive chunking, embedding + storage
│   ├── vectorstore.py   # ChromaDB wrapper (add, query, per-session collections)
│   ├── retrieval.py     # Embed query → top-k chunk retrieval
│   ├── generation.py    # Prompt templates + Groq API calls (chat & write modes)
│   └── requirements.txt
├── frontend/
│   ├── app.py           # Streamlit UI
│   └── requirements.txt
├── data/                # gitignored — holds uploads + ChromaDB persistence
│   ├── chroma/
│   └── uploads/
├── .env.example
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/haninnnnn/ScholarSync.git
cd ScholarSync
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Install frontend dependencies

```bash
pip install -r frontend/requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# Edit .env and set your GROQ_API_KEY
```

Get a free API key at [console.groq.com](https://console.groq.com/).

### 5. Start the FastAPI backend

```bash
cd backend
uvicorn main:app --reload --port 8000
```

The API docs are available at [http://localhost:8000/docs](http://localhost:8000/docs).

### 6. Start the Streamlit frontend (new terminal)

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🧪 Testing Individual Components

### Test PDF ingestion

```bash
cd backend
python ingest.py path/to/paper.pdf --session-id my_session
```

### Test vector store

```bash
cd backend
python vectorstore.py
```

### Test retrieval

```bash
cd backend
python retrieval.py "What methodology was used?" --session-id my_session
```

### Test full generation pipeline

```bash
cd backend
python generation.py "Summarize the key contributions" \
  --session-id my_session \
  --mode write \
  --write-mode literature_review
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Your Groq API key |
| `BACKEND_URL` | `http://localhost:8000` | Backend URL for the Streamlit frontend |

Chunking and retrieval parameters can be adjusted in their respective modules:
- `CHUNK_SIZE_WORDS = 400` and `CHUNK_OVERLAP_WORDS = 50` in `ingest.py`
- `DEFAULT_TOP_K = 5` in `retrieval.py`
- `GROQ_MODEL`, `MAX_TOKENS`, `TEMPERATURE` in `generation.py`

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| PDF Parsing | PyMuPDF (`fitz`) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persisted) |
| LLM | Groq API — Llama 3.3 70B Versatile |
| Frontend | Streamlit |
| Config | python-dotenv |

---

## 📝 API Reference

### `POST /upload`
Upload and ingest a PDF file.

**Form data:** `file` (PDF), `session_id` (optional, auto-generated if omitted)

**Response:**
```json
{
  "session_id": "abc-123",
  "title": "Attention Is All You Need",
  "pages_parsed": 15,
  "chunks_stored": 47,
  "message": "Successfully ingested ..."
}
```

### `POST /chat`
Ask a question grounded in uploaded papers.

```json
{
  "session_id": "abc-123",
  "message": "What attention mechanism does the paper propose?",
  "top_k": 5
}
```

### `POST /write`
Generate academic writing content.

```json
{
  "session_id": "abc-123",
  "prompt": "Focus on transformer applications in NLP",
  "write_mode": "literature_review",
  "top_k": 5
}
```

`write_mode` options: `summarize` | `compare` | `literature_review` | `citations`

---

## ⚠️ Limitations (v1)

- **No OCR:** Scanned/image-based PDFs are not supported. The PDF must contain selectable text.
- **In-memory chat history:** Conversation history resets when the backend restarts. Use a database for production.
- **Single-machine:** Embeddings run locally; a GPU speeds things up significantly for large batches.
- **Session isolation:** Each session is independent; no cross-session search.

---

## 📄 License

MIT — free to use, modify, and distribute.
