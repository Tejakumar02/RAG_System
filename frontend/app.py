"""
frontend/app.py — Streamlit UI for the RAG system.

Features:
  - Upload PDF / TXT / DOCX
  - Ask questions with streaming answers
  - View retrieved chunks with similarity scores + highlighting
  - Chat-style conversation memory
  - Document management (list / delete)
  - System health panel
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import httpx
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
SUPPORTED_TYPES = ["pdf", "txt", "docx"]

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Space Grotesk', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }
    
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .main-subtitle {
        color: #94a3b8;
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    .chunk-card {
        background: #0f172a;
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-left: 4px solid #6366f1;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #cbd5e1;
        line-height: 1.6;
        transition: border-color 0.2s;
    }
    
    .chunk-card:hover {
        border-left-color: #818cf8;
    }

    .chunk-meta {
        display: flex;
        gap: 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.7rem;
        color: #64748b;
    }

    .score-badge {
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        padding: 2px 8px;
        border-radius: 99px;
        font-weight: 600;
        font-size: 0.68rem;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .answer-box {
        background: linear-gradient(135deg, #0d1b2a, #1a2744);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        color: #e2e8f0;
        font-size: 0.95rem;
        line-height: 1.7;
        margin-top: 1rem;
    }

    .health-ok   { color: #4ade80; font-weight: 600; }
    .health-fail { color: #f87171; font-weight: 600; }

    .doc-badge {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 3px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    div[data-testid="stSidebar"] {
        background: #0a0a14;
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    div[data-testid="stSidebar"] label {
        color: #94a3b8 !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        font-family: 'Space Grotesk', sans-serif;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    .highlight { background: rgba(250, 204, 21, 0.18); border-radius: 3px; padding: 0 2px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────

def api_get(path: str, timeout: int = 10) -> Optional[dict]:
    try:
        r = httpx.get(f"{API_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, **kwargs) -> Optional[dict]:
    try:
        r = httpx.post(f"{API_URL}{path}", timeout=300, **kwargs)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', e.response.text)}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None
    except httpx.TimeoutException as e:
        st.error(f"Request timed out — Mistral is slow on first load. Try again in 10 seconds.")
        return None
    except httpx.HTTPStatusError as e:
        st.error(f"API error {e.response.status_code}: {e.response.json().get('detail', e.response.text)}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None


def highlight_text(text: str, keywords: List[str]) -> str:
    """Wrap matched keywords in highlight spans."""
    import re
    for kw in keywords:
        if len(kw) > 3:
            text = re.sub(
                f"({re.escape(kw)})",
                r'<span class="highlight">\1</span>',
                text,
                flags=re.IGNORECASE,
            )
    return text


# ── Session state init ────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []   # Chat history
if "doc_filter" not in st.session_state:
    st.session_state.doc_filter = []


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 DocMind RAG")
    st.markdown("---")

    # ── Health status
    st.markdown("### ⚡ System Status")
    health = api_get("/health")
    if health:
        ollama_ok = health.get("ollama_available", False)
        ollama_cls = "health-ok" if ollama_ok else "health-fail"
        ollama_text = "Online" if ollama_ok else "Offline"
        st.markdown(
            f"Ollama: <span class='{ollama_cls}'>{ollama_text}</span> "
            f"<code style='font-size:0.7rem'>{health.get('ollama_model','')}</code>",
            unsafe_allow_html=True,
        )
        st.metric("Indexed Chunks", health.get("total_chunks", 0))
        st.metric("Documents", health.get("documents_indexed", 0))
    else:
        st.warning("API unreachable — is the backend running?")

    st.markdown("---")

    # ── Upload
    st.markdown("### 📂 Upload Document")
    uploaded = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
    )
    overwrite = st.checkbox("Re-index if exists", value=False)

    if st.button("Upload & Index", use_container_width=True) and uploaded:
        with st.spinner(f"Indexing {uploaded.name}…"):
            result = api_post(
                "/upload_document",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                data={"overwrite": str(overwrite).lower()},
            )
        if result:
            status_val = result.get("status", "")
            if status_val == "skipped":
                st.info(result.get("message", "Already indexed."))
            else:
                st.success(
                    f"✅ Indexed **{result.get('doc_name')}**\n\n"
                    f"- Pages: {result.get('pages_extracted')}\n"
                    f"- Chunks: {result.get('chunks_created')}\n"
                    f"- Time: {result.get('ingestion_time_ms')}ms"
                )
            st.rerun()

    st.markdown("---")

    # ── Document list + filter
    st.markdown("### 📋 Indexed Documents")
    docs = api_get("/list_documents") or []

    if not docs:
        st.caption("No documents indexed yet.")
    else:
        all_names = [d["doc_name"] for d in docs]
        selected = st.multiselect(
            "Filter search to:",
            options=all_names,
            default=[],
            placeholder="All documents",
        )
        st.session_state.doc_filter = selected if selected else []

        for doc in docs:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(
                    f'<span class="doc-badge">📄</span> {doc["doc_name"]}<br>'
                    f'<small style="color:#64748b">{doc["chunk_count"]} chunks · {doc["page_count"]} pages</small>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑️", key=f"del_{doc['doc_name']}", help="Delete"):
                    api_post(f"/documents/{doc['doc_name']}", json={})
                    st.rerun()

    st.markdown("---")

    # ── Settings
    st.markdown("### ⚙️ Retrieval Settings")
    top_k = st.slider("Top-K chunks", 1, 10, 5)
    show_chunks = st.toggle("Show retrieved chunks", value=True)
    clear_chat = st.button("🧹 Clear chat", use_container_width=True)
    if clear_chat:
        st.session_state.messages = []
        st.rerun()


# ── Main panel ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <p class="main-title">🧠 DocMind — RAG Intelligence</p>
        <p class="main-subtitle">
            Upload documents · Ask questions · Get grounded, citation-backed answers
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab_chat, tab_docs, tab_arch = st.tabs(["💬 Chat", "📊 Documents", "🏗️ Architecture"])


# ── Chat tab ──────────────────────────────────────────────────────────────────
with tab_chat:
    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "chunks" in msg and show_chunks:
                with st.expander(
                    f"📎 {len(msg['chunks'])} retrieved chunks (click to expand)",
                    expanded=False,
                ):
                    for i, chunk in enumerate(msg["chunks"], 1):
                        keywords = msg.get("question", "").lower().split()
                        highlighted = highlight_text(chunk["text"], keywords)
                        score_pct = int(chunk["similarity_score"] * 100)
                        score_color = (
                            "#4ade80" if score_pct >= 75
                            else "#facc15" if score_pct >= 50
                            else "#f87171"
                        )
                        st.markdown(
                            f"""
                            <div class="chunk-card">
                                <div class="chunk-meta">
                                    <span>#{i}</span>
                                    <span>📄 {chunk['doc_name']}</span>
                                    <span>🔖 Page {chunk['page_number']}</span>
                                    <span class="score-badge" style="color:{score_color}">
                                        {score_pct}% match
                                    </span>
                                </div>
                                {highlighted}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents…"):
        if not docs:
            st.warning("⚠️ Upload a document first using the sidebar.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    t0 = time.time()
                    result = api_post(
                        "/query",
                        json={
                            "question": prompt,
                            "top_k": top_k,
                            "doc_filter": st.session_state.doc_filter or None,
                        },
                    )
                    elapsed = time.time() - t0

                if result:
                    answer = result.get("answer", "No answer generated.")
                    chunks = result.get("retrieved_chunks", [])

                    st.markdown(answer)
                    st.caption(
                        f"⏱ {result.get('latency_ms')}ms · "
                        f"🔍 {len(chunks)} chunks · "
                        f"🤖 {result.get('model')}"
                    )

                    if show_chunks and chunks:
                        with st.expander(
                            f"📎 {len(chunks)} retrieved chunks", expanded=False
                        ):
                            for i, chunk in enumerate(chunks, 1):
                                keywords = prompt.lower().split()
                                highlighted = highlight_text(chunk["text"], keywords)
                                score_pct = int(chunk["similarity_score"] * 100)
                                score_color = (
                                    "#4ade80" if score_pct >= 75
                                    else "#facc15" if score_pct >= 50
                                    else "#f87171"
                                )
                                st.markdown(
                                    f"""
                                    <div class="chunk-card">
                                        <div class="chunk-meta">
                                            <span>#{i}</span>
                                            <span>📄 {chunk['doc_name']}</span>
                                            <span>🔖 Page {chunk['page_number']}</span>
                                            <span class="score-badge" style="color:{score_color}">
                                                {score_pct}% match
                                            </span>
                                        </div>
                                        {highlighted}
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                    # Save to history
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "chunks": chunks,
                            "question": prompt,
                        }
                    )
                else:
                    st.error("Failed to get a response. Check API and Ollama status.")


# ── Documents tab ─────────────────────────────────────────────────────────────
with tab_docs:
    st.markdown("### 📊 Indexed Document Statistics")
    docs = api_get("/list_documents") or []

    if not docs:
        st.info("No documents indexed yet. Upload files using the sidebar.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Documents", len(docs))
        col2.metric("Total Chunks", sum(d["chunk_count"] for d in docs))
        col3.metric("Total Pages", sum(d["page_count"] for d in docs))

        st.markdown("---")
        for doc in docs:
            with st.expander(f"📄 {doc['doc_name']}", expanded=False):
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Chunks", doc["chunk_count"])
                col_b.metric("Pages", doc["page_count"])
                avg_chunk = doc["chunk_count"] / max(doc["page_count"], 1)
                col_c.metric("Avg Chunks/Page", f"{avg_chunk:.1f}")

                if st.button(f"Delete {doc['doc_name']}", key=f"del2_{doc['doc_name']}"):
                    r = httpx.delete(f"{API_URL}/documents/{doc['doc_name']}", timeout=10)
                    if r.status_code == 200:
                        st.success("Deleted.")
                        st.rerun()


# ── Architecture tab ──────────────────────────────────────────────────────────
with tab_arch:
    st.markdown("### 🏗️ System Architecture")
    st.code(
        """
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

  Data Flow (Query):
  ─────────────────
  User question
    → Embed question (all-MiniLM-L6-v2)
    → Cosine search in ChromaDB (Top-5)
    → Keyword rerank
    → Build grounded prompt
    → Ollama LLM generates answer
    → Return answer + chunk citations
        """,
        language="text",
    )
