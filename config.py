import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOADED_PDFS_DIR = DATA_DIR / "uploaded_pdfs"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOADED_PDFS_DIR.mkdir(parents=True, exist_ok=True)
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

# Chunking Configuration
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 120

# Embedding Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Retrieval Configuration
TOP_K_RETRIEVAL = 4

# LLM Configuration
DEFAULT_HF_MODEL = "google/flan-t5-base"
FALLBACK_FAST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# UI & Brand Configuration
APP_NAME = "DocPilot AI"
APP_SUBTITLE = "Intelligent AI Document Assistant"
APP_VERSION = "2.0.0"
APP_ICON = "⚡"

