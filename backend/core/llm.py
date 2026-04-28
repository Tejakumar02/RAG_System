"""
core/llm.py — LLM integration via Ollama.

The generation prompt is deliberately constrained:
  - Use ONLY the provided context
  - Explicitly say "I don't know" when context is insufficient
  - Never fabricate citations or facts

Supports streaming (for Streamlit) and full-response modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import httpx
from loguru import logger
from nltk import data
from tenacity import retry, stop_after_attempt, wait_exponential

from backend.config import settings
from backend.core.vector_store import RetrievedChunk


# ── Prompt template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, factual AI assistant powered by Retrieval-Augmented Generation.

STRICT RULES:
1. Answer ONLY using the provided context excerpts.
2. If the answer is not present in the context, respond exactly:
   "I don't have enough information in the provided documents to answer this question."
3. Never invent facts, statistics, dates, or names not explicitly in the context.
4. When quoting the context, reference the document name and page number.
5. Be concise but complete. Structure your answer clearly.
6. If multiple documents contain relevant information, synthesise them coherently."""


def build_prompt(question: str, chunks: List[RetrievedChunk]) -> str:
    """
    Construct the RAG prompt from retrieved chunks.

    Format:
        Context:
        [SOURCE: doc.pdf | Page 3 | Score: 0.87]
        <chunk text>

        ---
        Question: <user question>
    """
    if not chunks:
        context_block = "No relevant context found in the document store."
    else:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            header = (
                f"[EXCERPT {i} | Source: {chunk.doc_name} | "
                f"Page: {chunk.page_number} | "
                f"Relevance: {chunk.similarity_score:.0%}]"
            )
            context_parts.append(f"{header}\n{chunk.text}")
        context_block = "\n\n---\n\n".join(context_parts)

    return (
        f"Context:\n\n{context_block}\n\n"
        f"{'='*60}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


# ── Response model ────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    answer: str
    model: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


# ── Ollama client ─────────────────────────────────────────────────────────────

class OllamaLLM:
    """
    Thin wrapper around Ollama's /api/chat endpoint.
    Handles retries, timeouts, and both streaming/blocking modes.
    """

    def __init__(
        self,
        model: str = settings.LLM_MODEL,
        base_url: str = settings.OLLAMA_BASE_URL,
        temperature: float = settings.LLM_TEMPERATURE,
        max_tokens: int = settings.LLM_MAX_TOKENS,
        timeout: int = settings.LLM_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    # ── Health check ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception:
            return []

    # ── Generation ────────────────────────────────────────────────────────────

   
    
    def generate(self, question: str, chunks) -> LLMResponse:
        prompt = build_prompt(question, chunks)
        t0 = time.time()

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
        },
        "stream": False,
        }

    # Remove @retry decorator effect — direct call with explicit timeout
        with httpx.Client(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()

        data = resp.json()
        answer = data["message"]["content"].strip()
        latency_ms = int((time.time() - t0) * 1000)

        return LLMResponse(
        answer=answer,
        model=self.model,
        latency_ms=latency_ms,
    )

    def stream(
        self,
        question: str,
        chunks: List[RetrievedChunk],
    ) -> Generator[str, None, None]:
        """
        Streaming generation — yields text tokens as they arrive.
        Suitable for Streamlit's st.write_stream().
        """
        import json

        prompt = build_prompt(question, chunks)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            "stream": True,
        }

        with httpx.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk_data = json.loads(line)
                    token = chunk_data.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk_data.get("done", False):
                        break
                except json.JSONDecodeError:
                    continue
