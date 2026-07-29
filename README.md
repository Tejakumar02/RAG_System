# 🧠 DocMind Production RAG System

A complete **Retrieval-Augmented Generation** system built for production use.
Upload PDFs, TXT, or DOCX files and ask natural-language questions against them.
All answers are grounded in your documents — no hallucination.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DocMind RAG System                               │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────┐    Upload     ┌──────────────────────────────────────────┐
  │  User /  │──────────────▶│          FastAPI Backend                 │
  │Streamlit │               │  POST /upload_document                   │
  │   UI     │◀──────────────│  POST /query                             │
  └──────────┘    Answer     │  GET  /list_documents                    │
                             └──────────────┬───────────────────────────┘
                                            │
                          ┌─────────────────▼─────────────────┐
                          │         RAG Pipeline               │
                          │                                    │
                          │  1. DocumentIngester               │
                          │     ├── PDFExtractor (PyMuPDF)     │
                          │     ├── TXTExtractor               │
                          │     └── DOCXExtractor              │
                          │                                    │
                          │  2. TextCleaner                    │
                          │     └── Regex noise removal        │
                          │                                    │
                          │  3. SemanticChunker                │
                          │     ├── NLTK sentence splitting    │
                          │     ├── Token-bounded chunks       │
                          │     └── 15% overlap window        │
                          │                                    │
                          │  4. EmbeddingModel                 │
                          │     └── all-MiniLM-L6-v2 (384-d)  │
                          │                                    │
                          │  5. VectorStore (ChromaDB)         │
                          │     ├── Cosine similarity index    │
                          │     └── Metadata: doc, page, chunk │
                          │                                    │
                          │  6. Retrieval + Reranking          │
                          │     ├── Top-K cosine search        │
                          │     └── Keyword overlap boost      │
                          │                                    │
                          │  7. OllamaLLM (Mistral/LLaMA)     │
                          │     └── Grounded prompt template   │
                          └────────────────────────────────────┘
```

---

## 📁 Folder Structure

```
rag_system/
├── backend/
│   ├── __init__.py
│   ├── config.py               # All config / settings (Pydantic)
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py             # FastAPI app + all endpoints
│   └── core/
│       ├── __init__.py
│       ├── ingestion.py        # PDF/TXT/DOCX text extraction + cleaning
│       ├── chunker.py          # Semantic chunking with overlap
│       ├── embeddings.py       # Sentence-transformer embedding model
│       ├── vector_store.py     # ChromaDB persistence layer
│       ├── llm.py              # Ollama integration + prompt builder
│       └── rag_pipeline.py     # End-to-end orchestrator
├── frontend/
│   └── app.py                  # Streamlit UI
├── scripts/
│   └── setup.sh                # One-shot setup script
├── uploads/                    # Uploaded files (auto-created)
├── chroma_db/                  # ChromaDB persistence (auto-created)
├── logs/                       # Application logs (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- 4GB+ RAM (8GB recommended)

### 1. Clone & Install

```bash
git clone https://github.com/Tejakumar02/RAG_System.git
cd rag_system

# Install dependencies + download NLTK data
bash scripts/setup.sh
```

### 2. Copy environment config

```bash
cp .env.example .env
# Edit .env if needed (model, ports, etc.)
```

### 3. Start Ollama

```bash
# In a separate terminal
ollama serve

# Pull the model (first time only, ~4GB)
ollama pull mistral
```

### 4. Start the API

```bash
python -m uvicorn backend.api.main:app --reload --port 8000
```

### 5. Start the UI

```bash
streamlit run frontend/app.py
```

Open **http://localhost:8501** in your browser.

---

## 🔌 API Reference

### `POST /upload_document`
Upload and index a document.

```bash
curl -X POST http://localhost:8000/upload_document \
  -F "file=@my_document.pdf" \
  -F "overwrite=false"
```

Response:
```json
{
  "status": "success",
  "doc_name": "my_document.pdf",
  "file_type": "pdf",
  "pages_extracted": 12,
  "chunks_created": 47,
  "vectors_stored": 47,
  "ingestion_time_ms": 1842
}
```

### `POST /query`
Ask a question.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'
```

Response includes the grounded answer, model info, latency, and all retrieved chunks with similarity scores.

### `GET /list_documents`
List all indexed documents.

### `DELETE /documents/{doc_name}`
Remove a document and all its chunks.

### `GET /health`
System health check (Ollama status, total chunks, etc.).

---

## ⚙️ Configuration

Edit `.env` or modify `backend/config.py`:

| Parameter | Default | Description |
|---|---|---|
| `LLM_MODEL` | `mistral` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `CHUNK_SIZE` | `512` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `80` | Overlap tokens (~15%) |
| `TOP_K` | `5` | Chunks retrieved per query |
| `SIMILARITY_THRESHOLD` | `0.25` | Min cosine similarity to include |
| `LLM_TEMPERATURE` | `0.1` | Low = more deterministic |

---

## 🧠 Key Design Decisions

### Chunking
- **Sentence-aware**: NLTK Punkt tokenizer splits on sentence boundaries first, preserving linguistic coherence
- **Token-bounded**: tiktoken ensures chunks stay within the target window
- **15% overlap**: Every chunk shares ~80 tokens with its neighbour, preventing answer truncation at boundaries

### Embeddings
- `all-MiniLM-L6-v2`: 22M param model, 384-dim, excellent speed/quality tradeoff
- L2-normalised output → cosine similarity == dot product (fast)

### Grounding
- System prompt explicitly forbids fabrication
- "I don't know" response when context is missing
- Low temperature (0.1) for deterministic, factual outputs

### Reranking
- Lightweight keyword-overlap boost applied after cosine retrieval
- No extra model needed; adds ~1ms per query

---

## 🔥 Bonus Features Implemented

- ✅ Keyword highlighting in retrieved chunks (UI)
- ✅ Similarity score badges (color-coded: green/yellow/red)
- ✅ Chat-style conversation memory
- ✅ Document-level filter (query only specific files)
- ✅ Streaming-ready LLM client (for Streamlit integration)
- ✅ Health endpoint with Ollama status
- ✅ Automatic chunk overlap for context continuity
- ✅ Multi-extractor with fallback (PyMuPDF → pdfplumber)

---

## 📊 Performance

| Metric | Typical Value |
|---|---|
| Ingestion (10-page PDF) | 2–5s |
| Embedding (per chunk) | ~1ms on CPU |
| Retrieval (ChromaDB) | <50ms |
| LLM generation (Mistral) | 1–8s |
| **Total query latency** | **2–10s** |

For sub-2s responses: use `gemma2:2b` or `phi3:mini` in Ollama.











 🛡️ Production Hardening (Next Steps)

- [ ] Add API key authentication (`fastapi.security`)
- [ ] Rate limiting (`slowapi`)
- [ ] Async ingestion queue (Celery + Redis)
- [ ] Replace Ollama with vLLM for higher throughput
- [ ] Add evaluation framework (RAGAS)
- [ ] Docker Compose for one-command deployment
- [ ] Prometheus metrics endpoint
