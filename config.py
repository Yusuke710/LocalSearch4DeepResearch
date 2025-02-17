import os
from pathlib import Path
from enum import Enum
import secrets

class EmbeddingModel(Enum):
    SENTENCE_TRANSFORMER = "sentence-transformer"
    OPENAI = "openai"

# Directory structure
BASE_DIR = Path("data")
DOCS_DIR = BASE_DIR / "docs"
MARKDOWN_DIR = BASE_DIR / "processed" / "markdown"
EMBEDDINGS_DIR = BASE_DIR / "processed" / "embeddings"
INDEX_DIR = BASE_DIR / "processed" / "index"

# Configuration
EMBEDDING_MODEL = EmbeddingModel(os.getenv("EMBEDDING_MODEL", "openai"))
SENTENCE_TRANSFORMER_MODEL = os.getenv("SENTENCE_TRANSFORMER_MODEL", "all-MiniLM-L6-v2")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = 1536 if EMBEDDING_MODEL == EmbeddingModel.OPENAI else 384

# Markdown conversion configuration
USE_LLM_MARKDOWN = os.getenv("USE_LLM_MARKDOWN", "true").lower() == "true"

# API Configuration
API_KEY = os.getenv("LOCAL_API_KEY", secrets.token_urlsafe(32))
PORT = 5001

# Create directories if they don't exist
for dir_path in [DOCS_DIR, MARKDOWN_DIR, EMBEDDINGS_DIR, INDEX_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True) 