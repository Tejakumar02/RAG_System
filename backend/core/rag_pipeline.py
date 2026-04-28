"""
core/rag_pipeline.py — End-to-end RAG orchestrator.

Combines ingestion → chunking → embedding → retrieval → generation
into a single clean interface used by the API layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from loguru import logger

from backend.config import settings
from backend.core.chunker import SemanticChunker
from backend.core.embeddings import EmbeddingModel
from backend.core.ingestion import DocumentIngester
from backend.core.llm import LLMResponse, OllamaLLM
from backend.core.vector_store import RetrievedChunk, VectorStore


class RAGPipeline:
    """
    Singleton-style orchestrator for the full RAG pipeline.

    Can be instantiated once at app startup and reused across requests.
    """

    def __init__(self):
        logger.info("[RAG] Initialising pipeline components…")
        self.ingester = DocumentIngester()
        self.chunker = SemanticChunker()
        self.vector_store = VectorStore()
        self.llm = OllamaLLM()
        # Trigger embedding model warm-up
        _ = EmbeddingModel.get_instance()
        logger.info("[RAG] Pipeline ready.")

    # ── Document ingestion ────────────────────────────────────────────────────

    def ingest_document(self, file_path: Path, overwrite: bool = False) -> dict:
        """
        Full ingestion pipeline for a single document.

        Returns a summary dict with stats.
        """
        doc_name = file_path.name

        if self.vector_store.document_exists(doc_name):
            if not overwrite:
                logger.info(f"[RAG] '{doc_name}' already indexed. Skipping.")
                return {
                    "status": "skipped",
                    "doc_name": doc_name,
                    "message": "Document already indexed. Pass overwrite=True to re-index.",
                }
            else:
                deleted = self.vector_store.delete_document(doc_name)
                logger.info(f"[RAG] Deleted {deleted} old chunks for '{doc_name}'")

        # 1. Extract text
        doc_content = self.ingester.ingest(file_path)
        logger.info(f"[RAG] Extracted {doc_content.total_pages} pages")

        # 2. Chunk
        chunks = self.chunker.chunk_document(doc_content)
        logger.info(f"[RAG] Created {len(chunks)} chunks")

        # 3. Embed + store
        added = self.vector_store.add_chunks(chunks)
        logger.info(f"[RAG] Stored {added} vectors in ChromaDB")

        return {
            "status": "success",
            "doc_name": doc_name,
            "file_type": doc_content.file_type,
            "pages_extracted": doc_content.total_pages,
            "chunks_created": len(chunks),
            "vectors_stored": added,
        }

    # ── Query / Answer ────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        doc_filter: Optional[List[str]] = None,
    ) -> dict:
        """
        Full RAG query: retrieve → rerank → generate.

        Returns a dict with answer, retrieved chunks, and metadata.
        """
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant chunks
        chunks: List[RetrievedChunk] = self.vector_store.query(
            question=question,
            top_k=top_k,
            doc_filter=doc_filter,
        )
        logger.info(f"[RAG] Retrieved {len(chunks)} chunks for query")

        # 2. Optional simple reranker: boost chunks with exact keyword overlap
        chunks = self._rerank(question, chunks)

        # 3. Generate answer
        if not self.llm.is_available():
            raise RuntimeError(
                "Ollama is not reachable. "
                f"Ensure Ollama is running at {settings.OLLAMA_BASE_URL} "
                f"and model '{settings.LLM_MODEL}' is pulled."
            )

        llm_response: LLMResponse = self.llm.generate(
            question=question,
            chunks=chunks,
        )

        return {
            "answer": llm_response.answer,
            "model": llm_response.model,
            "latency_ms": llm_response.latency_ms,
            "chunks_retrieved": len(chunks),
            "retrieved_chunks": [
                {
                    "text": c.text,
                    "doc_name": c.doc_name,
                    "page_number": c.page_number,
                    "chunk_index": c.chunk_index,
                    "similarity_score": c.similarity_score,
                }
                for c in chunks
            ],
        }

    def get_retrieved_chunks(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        doc_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve chunks without generating an answer (for streaming UIs)."""
        chunks = self.vector_store.query(
            question=question, top_k=top_k, doc_filter=doc_filter
        )
        return self._rerank(question, chunks)

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> list:
        return self.vector_store.list_documents()

    def delete_document(self, doc_name: str) -> dict:
        count = self.vector_store.delete_document(doc_name)
        return {"deleted_chunks": count, "doc_name": doc_name}

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _rerank(question: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Simple keyword-overlap reranker.

        Boosts similarity score by 0–0.05 based on exact word overlap
        between query and chunk. Keeps chunks already sorted by cosine similarity
        but nudges semantically matched results higher.
        """
        if not chunks:
            return chunks

        q_words = set(question.lower().split())
        for chunk in chunks:
            chunk_words = set(chunk.text.lower().split())
            overlap = len(q_words & chunk_words)
            boost = min(overlap * 0.005, 0.05)  # Max +0.05
            chunk.similarity_score = min(chunk.similarity_score + boost, 1.0)

        chunks.sort(key=lambda c: c.similarity_score, reverse=True)
        return chunks
