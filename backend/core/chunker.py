"""
core/chunker.py — Semantic chunking with configurable overlap.

Strategy:
  1. Split text on sentence boundaries (NLTK Punkt tokeniser).
  2. Greedily accumulate sentences until chunk_size tokens reached.
  3. On each new chunk, seed it with the last `overlap` tokens
     from the previous chunk to preserve context continuity.
  4. Reject chunks smaller than MIN_CHUNK_SIZE.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

import tiktoken
from loguru import logger

from backend.config import settings
from backend.core.ingestion import DocumentContent, PageContent


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class TextChunk:
    """A single semantic chunk ready for embedding."""
    text: str
    chunk_index: int           # Global index within the document
    doc_name: str
    page_number: int
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_chroma_metadata(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "token_count": self.token_count,
            **self.metadata,
        }


# ── Tokenizer (cl100k_base ≈ all modern models) ───────────────────────────────

_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text))


def decode_tokens(tokens: list) -> str:
    return _TOKENIZER.decode(tokens)


# ── Sentence splitter ─────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    Uses NLTK Punkt if available, otherwise falls back to regex.
    """
    try:
        import nltk
        try:
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)
            tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentences = tokenizer.tokenize(text)
    except Exception:
        # Regex fallback: split on . ! ? followed by whitespace + capital
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

    return [s.strip() for s in sentences if s.strip()]


# ── Core chunker ──────────────────────────────────────────────────────────────

class SemanticChunker:
    """
    Produces overlapping token-bounded chunks from a DocumentContent.

    Parameters
    ----------
    chunk_size    : target maximum tokens per chunk
    chunk_overlap : tokens to carry over from previous chunk
    min_size      : minimum tokens; smaller chunks are discarded
    """

    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP,
        min_size: int = settings.MIN_CHUNK_SIZE,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_size = min_size

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk_document(self, doc: DocumentContent) -> List[TextChunk]:
        """Chunk all pages of a document, returning a flat list of TextChunks."""
        all_chunks: List[TextChunk] = []
        global_idx = 0

        for page in doc.pages:
            page_chunks = self._chunk_page(page, start_index=global_idx)
            all_chunks.extend(page_chunks)
            global_idx += len(page_chunks)

        logger.info(
            f"[Chunker] '{doc.filename}': "
            f"{doc.total_pages} pages → {len(all_chunks)} chunks "
            f"(size={self.chunk_size}, overlap={self.chunk_overlap})"
        )
        return all_chunks

    # ── Internal ──────────────────────────────────────────────────────────────

    def _chunk_page(self, page: PageContent, start_index: int) -> List[TextChunk]:
        sentences = split_sentences(page.text)
        if not sentences:
            return []

        chunks: List[TextChunk] = []
        current_sentences: List[str] = []
        current_tokens: List[int] = []   # flat token list for current chunk
        overlap_tokens: List[int] = []   # tokens to prepend from prev chunk

        def flush_chunk() -> Optional[TextChunk]:
            nonlocal current_sentences, current_tokens

            text = " ".join(current_sentences).strip()
            token_count = len(current_tokens)

            if token_count < self.min_size:
                return None

            chunk = TextChunk(
                text=text,
                chunk_index=start_index + len(chunks),
                doc_name=page.source_file,
                page_number=page.page_number,
                token_count=token_count,
                metadata=page.metadata.copy(),
            )
            return chunk

        for sentence in sentences:
            sent_tokens = _TOKENIZER.encode(sentence)

            # Single sentence exceeds chunk_size → split it hard
            if len(sent_tokens) > self.chunk_size:
                # Flush whatever we have
                if current_sentences:
                    c = flush_chunk()
                    if c:
                        chunks.append(c)
                    overlap_tokens = current_tokens[-self.chunk_overlap:]
                    current_sentences = []
                    current_tokens = list(overlap_tokens)

                # Split the giant sentence into sub-chunks
                for sub_chunk_tokens in self._split_token_list(
                    sent_tokens, self.chunk_size - len(overlap_tokens)
                ):
                    sub_text = _TOKENIZER.decode(sub_chunk_tokens)
                    sub_token_count = len(sub_chunk_tokens)
                    if sub_token_count >= self.min_size:
                        chunks.append(
                            TextChunk(
                                text=sub_text,
                                chunk_index=start_index + len(chunks),
                                doc_name=page.source_file,
                                page_number=page.page_number,
                                token_count=sub_token_count,
                                metadata=page.metadata.copy(),
                            )
                        )
                    overlap_tokens = sub_chunk_tokens[-self.chunk_overlap:]
                current_sentences = []
                current_tokens = list(overlap_tokens)
                continue

            # Would adding this sentence exceed the limit?
            if (len(current_tokens) + len(sent_tokens)) > self.chunk_size and current_sentences:
                c = flush_chunk()
                if c:
                    chunks.append(c)
                # Seed next chunk with overlap
                overlap_tokens = current_tokens[-self.chunk_overlap:]
                current_sentences = [_TOKENIZER.decode(overlap_tokens)] if overlap_tokens else []
                current_tokens = list(overlap_tokens)

            current_sentences.append(sentence)
            current_tokens.extend(sent_tokens)

        # Flush remaining
        if current_sentences:
            c = flush_chunk()
            if c:
                chunks.append(c)

        return chunks

    @staticmethod
    def _split_token_list(tokens: List[int], size: int) -> List[List[int]]:
        """Hard-split a token list into sub-lists of `size`."""
        return [tokens[i : i + size] for i in range(0, len(tokens), max(size, 1))]
