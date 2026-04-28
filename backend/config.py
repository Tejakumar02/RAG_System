"""
config.py — Centralized configuration for the RAG system.
All tunable parameters live here so you never hunt through code.
"""

from pydantic import Field
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # ── Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = Path(__file__).resolve().parent.parent / "uploads"
    CHROMA_DIR: Path = Path(__file__).resolve().parent.parent / "chroma_db"
    LOG_DIR: Path = Path(__file__).resolve().parent.parent / "logs"

    # ── Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 80
    MIN_CHUNK_SIZE: int = 50

    # ── Embedding
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"

    # ── ChromaDB
    CHROMA_COLLECTION: str = "rag_documents"

    # ── Retrieval
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25

    # ── LLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    LLM_MODEL: str = "mistral"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    LLM_TIMEOUT: int = 240   # hardcoded here as fallback

    # ── API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_WORKERS: int = 1
    ALLOWED_EXTENSIONS: set = {".pdf", ".txt", ".docx"}
    MAX_FILE_SIZE_MB: int = 50

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",       # ignore unknown keys in .env
    }

    def create_dirs(self):
        for d in [self.UPLOAD_DIR, self.CHROMA_DIR, self.LOG_DIR]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_dirs()