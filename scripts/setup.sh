#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

# ── Colors ──────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "  ${GREEN}==${NC} $*"; }
warn()  { echo -e "  ${YELLOW}==${NC} $*"; }
fail()  { echo -e "  ${RED}==${NC} $*"; exit 1; }
header(){ echo -e "\n${CYAN}$*${NC}"; }

# ── 1. Prerequisites ────────────────────────────
header "▸ Checking prerequisites"

if ! command -v docker &>/dev/null; then
  fail "Docker not found. See https://docs.docker.com/engine/install/"
fi
info "docker found: $(docker --version 2>/dev/null)"

# ── 2. Create .env if missing ───────────────────
header "▸ Configuration"

if [ ! -f ".env" ]; then
  cp ".env.example" ".env"
  info "Created .env from .env.example"
else
  warn ".env already exists — will not overwrite"
fi

# ── 3. Provider picker ──────────────────────────
echo ""
echo -e "  ${CYAN}Which LLM / embedding provider do you want to use?${NC}"
echo "    1) Local (LM Studio / Ollama) -- default, works offline"
echo "    2) OpenAI API"
echo "    3) Groq"
echo "    4) Other OpenAI-compatible API"
echo "    5) Skip -- I will edit .env myself"
read -r -p "  Enter 1-5 (default: 1): " choice
choice="${choice:-1}"

case "$choice" in
  1)
    info "Configuring for local provider"
    llm_url="http://host.docker.internal:1234/v1"
    emb_url="http://host.docker.internal:1234/v1"
    model="google/gemma-4-e4b"
    emb_model="text-embedding-nomic-embed-text-v1.5@q8_0"
    emb_dims="768"
    api_key="not-needed"

    read -r -p "  Local type? (1) LM Studio (default) / (2) Ollama: " local_type
    if [ "$local_type" = "2" ]; then
      llm_url="http://host.docker.internal:11434/v1"
      emb_url="http://host.docker.internal:11434/v1"
      model="llama3.2"
      emb_model="nomic-embed-text"
      emb_dims="768"
    fi
    ;;
  2)
    info "Configuring for OpenAI"
    read -r -p "  Enter your OpenAI API key (sk-...): " api_key
    read -r -p "  LLM model (default: gpt-4o-mini): " model
    model="${model:-gpt-4o-mini}"
    read -r -p "  Embedding model (default: text-embedding-3-small): " emb_model
    emb_model="${emb_model:-text-embedding-3-small}"
    read -r -p "  Embedding dimensions (1536 for 3-small, default: 1536): " emb_dims
    emb_dims="${emb_dims:-1536}"
    llm_url="https://api.openai.com/v1"
    emb_url="https://api.openai.com/v1"
    ;;
  3)
    info "Configuring for Groq"
    read -r -p "  Enter your Groq API key (gsk_...): " api_key
    read -r -p "  LLM model (default: llama-3.3-70b-versatile): " model
    model="${model:-llama-3.3-70b-versatile}"
    read -r -p "  Embedding model (default: nomic-embed-text-v1.5): " emb_model
    emb_model="${emb_model:-nomic-embed-text-v1.5}"
    read -r -p "  Embedding dimensions (default: 768): " emb_dims
    emb_dims="${emb_dims:-768}"
    llm_url="https://api.groq.com/openai/v1"
    emb_url="https://api.groq.com/openai/v1"
    ;;
  4)
    info "Configuring custom OpenAI-compatible API"
    read -r -p "  LLM API base URL (e.g. https://api.openai.com/v1): " llm_url
    read -r -p "  Embedding API base URL: " emb_url
    read -r -p "  API key: " api_key
    read -r -p "  LLM model: " model
    read -r -p "  Embedding model: " emb_model
    read -r -p "  Embedding dimensions: " emb_dims
    ;;
  *)
    warn "Skipping auto-config — edit .env manually then re-run"
    ;;
esac

# ── 4. Write provider settings to .env ──────────
if [ "$choice" != "5" ]; then
  # Remove old OPENAI_BASE_URL, EMBEDDING_BASE_URL, LLM_BASE_URL, LLM_API_KEY
  sed -i '/^OPENAI_BASE_URL=/d' .env
  sed -i '/^EMBEDDING_BASE_URL=/d' .env
  sed -i '/^LLM_BASE_URL=/d' .env
  sed -i '/^LLM_API_KEY=/d' .env

  # Update key-value pairs
  sed -i "s|^OPENAI_API_KEY=not-needed|OPENAI_API_KEY=$api_key|" .env
  sed -i "s|^EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5@q8_0|EMBEDDING_MODEL=$emb_model|" .env
  sed -i "s|^EMBEDDING_DIMENSIONS=768|EMBEDDING_DIMENSIONS=$emb_dims|" .env
  sed -i "s|^LLM_MODEL=google/gemma-4-e4b|LLM_MODEL=$model|" .env

  # Append provider URLs
  cat >> .env << EOF

# -- Provider URLs (set by setup script) --
EMBEDDING_BASE_URL=$emb_url
LLM_BASE_URL=$llm_url
EOF

  if [ "$api_key" != "not-needed" ] && [ -n "$api_key" ]; then
    echo "LLM_API_KEY=$api_key" >> .env
  fi

  info ".env updated"
fi

# ── 5. Start Docker containers ──────────────────
header "▸ Starting containers"
info "Running: docker compose up -d"
docker compose up -d
if [ $? -ne 0 ]; then fail "docker compose failed"; fi

# ── 6. Wait for health ──────────────────────────
header "▸ Waiting for service to be ready"
for i in $(seq 1 30); do
  status=$(curl -sf http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
  if [ "$status" = "ok" ]; then
    count=$(curl -sf http://localhost:8000/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('memory_count','?'))" 2>/dev/null || echo "?")
    info "memory-agent is running! Memories stored: $count"
    break
  fi
  if [ "$i" -eq 30 ]; then
    warn "Health check not responding after 30 seconds — check 'docker compose logs app'"
  else
    echo -ne "   Waiting... ($i/30)\r"
    sleep 2
  fi
done

# ── 7. Harness configs ──────────────────────────
header "▸ Setup complete"
echo -e "  ${GREEN}Memory agent is running at http://localhost:8000${NC}"
echo -e "  ${GREEN}Database: PostgreSQL 16 + pgvector on port 5433${NC}"

echo ""
echo -e "  ${CYAN}── MCP Config ─────────────────────────────${NC}"
echo "  Add to your AI tool's MCP settings:"
echo ""
echo -e "  ${YELLOW}opencode.json / .mcp.json:${NC}"
echo '  {'
echo '    "mcpServers": {'
echo '      "memory": {'
echo '        "command": "docker",'
echo '        "args": ["exec", "-i", "memory-agent-app", "python", "memory_mcp.py"]'
echo '      }'
echo '    }'
echo '  }'
echo ""
echo -e "  ${YELLOW}Or use the HTTP API directly:${NC}"
echo '  curl http://localhost:8000/health'
echo '  curl -X POST http://localhost:8000/add -H "Content-Type: application/json" -d "{...}"'
echo ""
echo -e "  ${CYAN}── Quick test ──────────────────────────────${NC}"
echo '  curl -s http://localhost:8000/health'
