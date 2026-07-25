from dotenv import load_dotenv
import os
from pathlib import Path

# Load .env from project root (two levels up from this file if in src/, or CWD)
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if not dotenv_path.exists():
    dotenv_path = Path.cwd() / ".env"
load_dotenv(dotenv_path)


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:localdev@localhost:5433/memory_agent",
)

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-nomic-embed-text-v1.5@q8_0")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", OPENAI_BASE_URL)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "google/gemma-4-e4b")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", OPENAI_API_KEY)
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

EXTRACTION_EVERY_N_TURNS = int(os.getenv("EXTRACTION_EVERY_N_TURNS", "5"))
EXTRACTION_MAX_MEMORIES = int(os.getenv("EXTRACTION_MAX_MEMORIES", "20"))
PERSONA_EVERY_N_MEMORIES = int(os.getenv("PERSONA_EVERY_N_MEMORIES", "50"))

RECALL_STRATEGY = os.getenv("RECALL_STRATEGY", "hybrid")
RECALL_MAX_RESULTS = int(os.getenv("RECALL_MAX_RESULTS", "10"))
RECALL_SIMILARITY_THRESHOLD = float(os.getenv("RECALL_SIMILARITY_THRESHOLD", "0.3"))
# Unused with RRF (rank-based fusion); kept for env compat only.
RECALL_KEYWORD_WEIGHT = float(os.getenv("RECALL_KEYWORD_WEIGHT", "0.3"))
RECALL_RRF_K = int(os.getenv("RECALL_RRF_K", "60"))
