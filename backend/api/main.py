"""
api/main.py — FastAPI application with full RAG endpoints.

Endpoints:
  POST  /upload_document   — Upload and index a document
  POST  /query             — Ask a question against indexed documents
  GET   /list_documents    — List all indexed documents
  DELETE /documents/{name} — Remove a document
  GET   /health            — System health check
"""

from __future__ import annotations

import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from backend.config import settings
from backend.core.rag_pipeline import RAGPipeline


# ── App lifecycle ─────────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the RAG pipeline once at startup."""
    global _pipeline
    logger.info("🚀 Starting RAG API server…")
    _pipeline = RAGPipeline()
    logger.info("✅ RAG pipeline ready.")
    yield
    logger.info("👋 Shutting down RAG API server.")


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return _pipeline


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Production RAG API",
    description="Retrieval-Augmented Generation over your documents",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=10)
    doc_filter: Optional[List[str]] = Field(
        default=None,
        description="Restrict search to these document names"
    )


class ChunkResult(BaseModel):
    text: str
    doc_name: str
    page_number: int
    chunk_index: int
    similarity_score: float


class QueryResponse(BaseModel):
    answer: str
    model: str
    latency_ms: int
    chunks_retrieved: int
    retrieved_chunks: List[ChunkResult]
    question: str


class DocumentInfo(BaseModel):
    doc_name: str
    chunk_count: int
    page_count: int


class UploadResponse(BaseModel):
    status: str
    doc_name: str
    file_type: str
    pages_extracted: int
    chunks_created: int
    vectors_stored: int
    ingestion_time_ms: int


class HealthResponse(BaseModel):
    status: str
    ollama_available: bool
    ollama_model: str
    total_chunks: int
    documents_indexed: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def validate_file(file: UploadFile) -> None:
    suffix = Path(file.filename).suffix.lower()
    if suffix not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{suffix}' not supported. "
                f"Allowed: {sorted(settings.ALLOWED_EXTENSIONS)}"
            ),
        )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(pipeline: RAGPipeline = Depends(get_pipeline)):
    """Check system status including Ollama availability."""
    docs = pipeline.list_documents()
    return HealthResponse(
        status="healthy",
        ollama_available=pipeline.llm.is_available(),
        ollama_model=pipeline.llm.model,
        total_chunks=pipeline.vector_store.total_chunks(),
        documents_indexed=len(docs),
    )


@app.post(
    "/upload_document",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(...),
    overwrite: bool = Form(default=False),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Upload and index a document (PDF, TXT, DOCX).

    - Extracts and cleans text
    - Splits into semantic chunks
    - Embeds and stores in ChromaDB
    """
    validate_file(file)

    # Save upload to disk
    save_path = settings.UPLOAD_DIR / file.filename
    t0 = time.time()

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        # Check file size
        size_mb = save_path.stat().st_size / (1024 * 1024)
        if size_mb > settings.MAX_FILE_SIZE_MB:
            save_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large ({size_mb:.1f}MB). Max: {settings.MAX_FILE_SIZE_MB}MB",
            )

        result = pipeline.ingest_document(save_path, overwrite=overwrite)
        ingestion_ms = int((time.time() - t0) * 1000)

        if result["status"] == "skipped":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={**result, "ingestion_time_ms": ingestion_ms},
            )

        return UploadResponse(
            ingestion_time_ms=ingestion_ms,
            **{k: result[k] for k in UploadResponse.model_fields if k in result},
        )

    except (ValueError, RuntimeError) as e:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        save_path.unlink(missing_ok=True)
        logger.exception(f"Upload failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Ask a question against indexed documents.

    Returns the grounded answer + the retrieved context chunks with scores.
    """
    if pipeline.vector_store.total_chunks() == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No documents indexed yet. Upload documents first.",
        )

    try:
        result = pipeline.query(
            question=request.question,
            top_k=request.top_k,
            doc_filter=request.doc_filter,
        )
        return QueryResponse(question=request.question, **result)

    except TimeoutError as e:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    except Exception as e:
        logger.exception(f"Query failed: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/list_documents", response_model=List[DocumentInfo], tags=["Documents"])
async def list_documents(pipeline: RAGPipeline = Depends(get_pipeline)):
    """List all indexed documents with chunk and page counts."""
    return pipeline.list_documents()


@app.delete("/documents/{doc_name}", tags=["Documents"])
async def delete_document(
    doc_name: str,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """Remove a document and all its chunks from the vector store."""
    result = pipeline.delete_document(doc_name)
    if result["deleted_chunks"] == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{doc_name}' not found.",
        )
    return result


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=True,
        log_level="info",
    )
