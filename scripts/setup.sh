#!/usr/bin/env bash
# scripts/setup.sh — One-shot environment setup for the RAG system
set -e

echo "════════════════════════════════════════════════════"
echo "  DocMind RAG — Environment Setup"
echo "════════════════════════════════════════════════════"

# 1. Python deps
echo ""
echo "▶ Installing Python dependencies…"
pip install -r requirements.txt --quiet

# 2. NLTK data
echo ""
echo "▶ Downloading NLTK punkt tokenizer…"
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# 3. Check Ollama
echo ""
echo "▶ Checking Ollama…"
if command -v ollama &>/dev/null; then
    echo "  ✅ Ollama found"
    echo "  Pulling mistral model (may take a few minutes on first run)…"
    ollama pull mistral
else
    echo "  ⚠️  Ollama not found. Install from https://ollama.com"
    echo "     Then run: ollama pull mistral"
fi

echo ""
echo "════════════════════════════════════════════════════"
echo "  Setup complete! Run the app:"
echo ""
echo "  Terminal 1 — Backend:"
echo "    cd rag_system"
echo "    python -m uvicorn backend.api.main:app --reload --port 8000"
echo ""
echo "  Terminal 2 — Frontend:"
echo "    cd rag_system"
echo "    streamlit run frontend/app.py"
echo ""
echo "  Then open: http://localhost:8501"
echo "════════════════════════════════════════════════════"
