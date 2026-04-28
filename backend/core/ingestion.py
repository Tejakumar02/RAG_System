"""
core/ingestion.py — Document ingestion pipeline.

Supports PDF, TXT, DOCX.
Extracts text, cleans noise, returns structured page data.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from loguru import logger


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class PageContent:
    """Represents extracted content from a single page / section."""
    text: str
    page_number: int
    source_file: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DocumentContent:
    """Full extracted content from a document."""
    filename: str
    file_type: str
    pages: List[PageContent]
    total_pages: int

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


# ── Text cleaner ──────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Normalises raw extracted text:
    - Unicode normalisation (NFC)
    - Remove control characters
    - Collapse whitespace
    - Remove common header/footer patterns
    - Strip repeated punctuation
    """

    # Patterns that are almost never content
    NOISE_PATTERNS = [
        r"Page\s+\d+\s+of\s+\d+",          # "Page 1 of 10"
        r"^\s*\d+\s*$",                      # Lone page numbers
        r"©.*?(?:\d{4})",                    # Copyright lines
        r"All rights reserved\.?",
        r"Confidential.*?(?:\n|$)",
        r"www\.[^\s]+",                      # Bare URLs (optional)
        r"-{3,}",                            # Horizontal rules
        r"_{3,}",
    ]

    def __init__(self):
        self._noise_re = re.compile(
            "|".join(self.NOISE_PATTERNS), re.IGNORECASE | re.MULTILINE
        )

    def clean(self, text: str) -> str:
        if not text:
            return ""
        # Unicode normalise
        text = unicodedata.normalize("NFC", text)
        # Remove control chars except newline/tab
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]", "", text)
        # Remove noise patterns
        text = self._noise_re.sub(" ", text)
        # Collapse multiple spaces (not newlines)
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ consecutive newlines → 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Strip leading/trailing whitespace per line
        lines = [line.strip() for line in text.splitlines()]
        # Drop blank-only lines at the very start/end
        text = "\n".join(lines).strip()
        return text


# ── Extractors ────────────────────────────────────────────────────────────────

class PDFExtractor:
    """Extracts text from PDF using PyMuPDF (fitz). Falls back to pdfplumber."""

    def extract(self, path: Path) -> List[PageContent]:
        try:
            return self._extract_pymupdf(path)
        except Exception as e:
            logger.warning(f"PyMuPDF failed ({e}), trying pdfplumber…")
            return self._extract_pdfplumber(path)

    def _extract_pymupdf(self, path: Path) -> List[PageContent]:
        import fitz  # PyMuPDF

        cleaner = TextCleaner()
        pages: List[PageContent] = []

        with fitz.open(str(path)) as doc:
            for page_num, page in enumerate(doc, start=1):
                raw = page.get_text("text")
                cleaned = cleaner.clean(raw)
                if len(cleaned.split()) < 5:
                    continue  # Skip near-empty pages
                pages.append(
                    PageContent(
                        text=cleaned,
                        page_number=page_num,
                        source_file=path.name,
                        metadata={"extractor": "pymupdf"},
                    )
                )

        logger.info(f"[PDF] Extracted {len(pages)} pages from '{path.name}'")
        return pages

    def _extract_pdfplumber(self, path: Path) -> List[PageContent]:
        import pdfplumber

        cleaner = TextCleaner()
        pages: List[PageContent] = []

        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text() or ""
                cleaned = cleaner.clean(raw)
                if len(cleaned.split()) < 5:
                    continue
                pages.append(
                    PageContent(
                        text=cleaned,
                        page_number=page_num,
                        source_file=path.name,
                        metadata={"extractor": "pdfplumber"},
                    )
                )

        logger.info(f"[PDF/plumber] Extracted {len(pages)} pages from '{path.name}'")
        return pages


class TXTExtractor:
    """Extracts plain text, splitting into pseudo-pages of ~50 lines."""

    LINES_PER_PAGE = 50

    def extract(self, path: Path) -> List[PageContent]:
        cleaner = TextCleaner()
        raw = path.read_text(encoding="utf-8", errors="replace")
        cleaned = cleaner.clean(raw)
        lines = cleaned.splitlines()

        pages: List[PageContent] = []
        for i in range(0, max(len(lines), 1), self.LINES_PER_PAGE):
            chunk_lines = lines[i : i + self.LINES_PER_PAGE]
            text = "\n".join(chunk_lines).strip()
            if text:
                pages.append(
                    PageContent(
                        text=text,
                        page_number=(i // self.LINES_PER_PAGE) + 1,
                        source_file=path.name,
                        metadata={"extractor": "txt"},
                    )
                )

        logger.info(f"[TXT] Extracted {len(pages)} sections from '{path.name}'")
        return pages


class DOCXExtractor:
    """Extracts text from DOCX preserving paragraph structure."""

    def extract(self, path: Path) -> List[PageContent]:
        from docx import Document

        cleaner = TextCleaner()
        doc = Document(str(path))

        # Group paragraphs into pseudo-pages (~30 paragraphs each)
        PARAS_PER_PAGE = 30
        all_paras = [p.text for p in doc.paragraphs if p.text.strip()]

        pages: List[PageContent] = []
        for i in range(0, max(len(all_paras), 1), PARAS_PER_PAGE):
            chunk = all_paras[i : i + PARAS_PER_PAGE]
            text = cleaner.clean("\n".join(chunk))
            if text:
                pages.append(
                    PageContent(
                        text=text,
                        page_number=(i // PARAS_PER_PAGE) + 1,
                        source_file=path.name,
                        metadata={"extractor": "docx"},
                    )
                )

        logger.info(f"[DOCX] Extracted {len(pages)} sections from '{path.name}'")
        return pages


# ── Main ingestion facade ─────────────────────────────────────────────────────

class DocumentIngester:
    """
    Single entry point for document ingestion.
    Routes to the correct extractor based on file extension.
    """

    EXTRACTORS = {
        ".pdf": PDFExtractor,
        ".txt": TXTExtractor,
        ".docx": DOCXExtractor,
    }

    def ingest(self, path: Path) -> DocumentContent:
        suffix = path.suffix.lower()
        extractor_cls = self.EXTRACTORS.get(suffix)

        if extractor_cls is None:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {list(self.EXTRACTORS)}"
            )

        extractor = extractor_cls()
        pages = extractor.extract(path)

        if not pages:
            raise RuntimeError(
                f"No text could be extracted from '{path.name}'. "
                "The file may be scanned/image-only."
            )

        return DocumentContent(
            filename=path.name,
            file_type=suffix.lstrip("."),
            pages=pages,
            total_pages=len(pages),
        )
