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
# Defaults match the compose TEI sidecar (MiniLM 384-d). Override via env if needed.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "384"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "not-needed")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1")
EMBEDDING_BASE_URL = (os.getenv("EMBEDDING_BASE_URL") or OPENAI_BASE_URL).rstrip("/")
# OpenAI text-embedding-3-* accepts dimensions=; TEI/MiniLM reject it.
EMBEDDING_SEND_DIMENSIONS = os.getenv("EMBEDDING_SEND_DIMENSIONS", "false").lower() in (
    "1",
    "true",
    "yes",
)

# LLM chat — read only these from the environment (.env via compose env_file).
# No rewriting, no provider maps, no fallback to embedding/OpenAI base.
LLM_PROVIDER = (os.getenv("LLM_PROVIDER") or "").strip()
LLM_MODEL = (os.getenv("LLM_MODEL") or "").strip()
LLM_API_KEY = (os.getenv("LLM_API_KEY") or "").strip()
LLM_BASE_URL = (os.getenv("LLM_BASE_URL") or "").strip().rstrip("/")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))

EXTRACTION_EVERY_N_TURNS = int(os.getenv("EXTRACTION_EVERY_N_TURNS", "5"))
EXTRACTION_MAX_MEMORIES = int(os.getenv("EXTRACTION_MAX_MEMORIES", "20"))
# Wall-clock: L0 newer than last extract + lag over this ⇒ recall untrusted (0 disables).
EXTRACTION_MAX_LAG_SECONDS = int(os.getenv("EXTRACTION_MAX_LAG_SECONDS", "3600"))
PERSONA_EVERY_N_MEMORIES = int(os.getenv("PERSONA_EVERY_N_MEMORIES", "50"))

RECALL_STRATEGY = os.getenv("RECALL_STRATEGY", "hybrid")
RECALL_MAX_RESULTS = int(os.getenv("RECALL_MAX_RESULTS", "10"))
RECALL_SIMILARITY_THRESHOLD = float(os.getenv("RECALL_SIMILARITY_THRESHOLD", "0.3"))
# Unused with RRF (rank-based fusion); kept for env compat only.
RECALL_KEYWORD_WEIGHT = float(os.getenv("RECALL_KEYWORD_WEIGHT", "0.3"))
RECALL_RRF_K = int(os.getenv("RECALL_RRF_K", "60"))

# Door: shared secret for HTTP (header X-Memory-Key). Empty = auth off (local tests).
MEMORY_API_SECRET = (os.getenv("MEMORY_API_SECRET") or "").strip()
# Hot-reload LLM config is off unless explicitly enabled.
MEMORY_ALLOW_RELOAD = os.getenv("MEMORY_ALLOW_RELOAD", "false").lower() in (
    "1",
    "true",
    "yes",
)
# Host bind for bare uvicorn (__main__). Compose publishes 127.0.0.1; in-container stays 0.0.0.0.
MEMORY_BIND_HOST = (os.getenv("MEMORY_BIND_HOST") or "127.0.0.1").strip() or "127.0.0.1"
