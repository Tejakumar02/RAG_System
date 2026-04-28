"""
core/vector_store.py — ChromaDB persistence layer.

Responsibilities:
  - Initialise / connect to ChromaDB collection
  - Upsert embedded chunks with metadata
  - Cosine similarity retrieval (top-k)
  - Document management (list, delete)

ChromaDB uses cosine distance internally when the collection is created
with cosine space; we store L2-normalised embeddings so that
cosine similarity = dot product, giving us an easy similarity score.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from loguru import logger

from backend.config import settings
from backend.core.chunker import TextChunk
from backend.core.embeddings import EmbeddingModel


# ── Retrieved chunk ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    text: str
    doc_name: str
    page_number: int
    chunk_index: int
    similarity_score: float   # 0.0 – 1.0 (higher = more relevant)
    metadata: Dict[str, Any]


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStore:
    """
    Wraps a ChromaDB persistent client and a single collection.
    One collection stores all documents; documents are filtered
    via metadata (doc_name).
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(settings.CHROMA_DIR),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION,
            metadata={"hnsw:space": "cosine"},   # Use cosine distance
        )
        self._embedder = EmbeddingModel.get_instance()
        logger.info(
            f"[VectorStore] Collection '{settings.CHROMA_COLLECTION}' ready. "
            f"Total vectors: {self._collection.count()}"
        )

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Embed and upsert a list of TextChunks.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [c.to_chroma_metadata() for c in chunks]

        self._collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

        logger.info(
            f"[VectorStore] Upserted {len(chunks)} chunks "
            f"from '{chunks[0].doc_name}'"
        )
        return len(chunks)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: int = settings.TOP_K,
        doc_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve the top-k most relevant chunks for a question.

        Parameters
        ----------
        question   : Natural-language user query
        top_k      : Number of results to return
        doc_filter : Optional list of doc_names to restrict search to
        """
        query_embedding = self._embedder.embed_single(question)

        where_clause = None
        if doc_filter:
            if len(doc_filter) == 1:
                where_clause = {"doc_name": {"$eq": doc_filter[0]}}
            else:
                where_clause = {"doc_name": {"$in": doc_filter}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        retrieved: List[RetrievedChunk] = []

        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # ChromaDB cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
            similarity = 1.0 - dist

            if similarity < settings.SIMILARITY_THRESHOLD:
                continue

            retrieved.append(
                RetrievedChunk(
                    text=text,
                    doc_name=meta.get("doc_name", "unknown"),
                    page_number=int(meta.get("page_number", 0)),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    similarity_score=round(similarity, 4),
                    metadata=meta,
                )
            )

        # Sort descending by similarity
        retrieved.sort(key=lambda c: c.similarity_score, reverse=True)
        logger.debug(
            f"[VectorStore] Query retrieved {len(retrieved)} chunks "
            f"(threshold={settings.SIMILARITY_THRESHOLD})"
        )
        return retrieved

    # ── Document management ───────────────────────────────────────────────────

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return unique documents stored with chunk counts."""
        if self._collection.count() == 0:
            return []

        all_meta = self._collection.get(include=["metadatas"])["metadatas"]

        doc_map: Dict[str, Dict[str, Any]] = {}
        for meta in all_meta:
            name = meta.get("doc_name", "unknown")
            if name not in doc_map:
                doc_map[name] = {"doc_name": name, "chunk_count": 0, "pages": set()}
            doc_map[name]["chunk_count"] += 1
            doc_map[name]["pages"].add(meta.get("page_number", 0))

        return [
            {
                "doc_name": v["doc_name"],
                "chunk_count": v["chunk_count"],
                "page_count": len(v["pages"]),
            }
            for v in doc_map.values()
        ]

    def delete_document(self, doc_name: str) -> int:
        """Delete all chunks for a document. Returns number deleted."""
        ids_result = self._collection.get(
            where={"doc_name": {"$eq": doc_name}}, include=[]
        )
        ids = ids_result.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
            logger.info(f"[VectorStore] Deleted {len(ids)} chunks for '{doc_name}'")
        return len(ids)

    def document_exists(self, doc_name: str) -> bool:
        result = self._collection.get(
            where={"doc_name": {"$eq": doc_name}}, include=[], limit=1
        )
        return len(result.get("ids", [])) > 0

    def total_chunks(self) -> int:
        return self._collection.count()
