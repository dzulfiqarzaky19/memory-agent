#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/setup.sh
#   ./scripts/setup.sh --claude        # also install Claude Code plugin
#   ./scripts/setup.sh --claude-only   # plugin only

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

CLAUDE=0
CLAUDE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --claude) CLAUDE=1 ;;
    --claude-only) CLAUDE_ONLY=1 ;;
    -h|--help)
      echo "Usage: $0 [--claude] [--claude-only]"
      exit 0
      ;;
  esac
done

# ── Colors ──────────────────────────────────────
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "  ${GREEN}==${NC} $*"; }
warn()  { echo -e "  ${YELLOW}==${NC} $*"; }
fail()  { echo -e "  ${RED}==${NC} $*"; exit 1; }
header(){ echo -e "\n${CYAN}$*${NC}"; }

install_claude_plugin() {
  header "▸ Claude Code plugin (auto-capture)"
  local integrations="$REPO_DIR/integrations"
  local plugin_dir="$integrations/claude-code"
  if [ ! -f "$plugin_dir/.claude-plugin/plugin.json" ]; then
    fail "Plugin missing at $plugin_dir"
  fi
  if ! command -v claude &>/dev/null; then
    warn "claude CLI not on PATH — wire manually:"
    echo "  claude plugin marketplace add \"$integrations\""
    echo "  claude plugin install memory-agent@memory-agent-integrations -s user"
    return 0
  fi
  info "Validating manifests"
  claude plugin validate "$plugin_dir" || true
  claude plugin validate "$integrations" || true
  info "Adding local marketplace"
  claude plugin marketplace add "$integrations" || true
  info "Installing memory-agent@memory-agent-integrations"
  if claude plugin install "memory-agent@memory-agent-integrations" -s user; then
    info "Plugin installed. Restart Claude Code / open a new session."
  else
    warn "plugin install failed — retry: claude plugin install memory-agent@memory-agent-integrations -s user"
  fi
  echo ""
  echo "  Optional env: MEMORY_AGENT_URL MEMORY_AGENT_USER_ID MEMORY_API_SECRET"
  echo "  (or ~/.memory-agent/api-secret for the Stop hook)"
  echo "  Capture log: ~/.claude/hooks/logs/memory-auto-capture.jsonl"
}

if [ "$CLAUDE_ONLY" -eq 1 ]; then
  install_claude_plugin
  exit 0
fi

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

# Door: MEMORY_API_SECRET in .env + ~/.memory-agent/api-secret (value never printed).
header "▸ API door (localhost + shared secret)"
door_out="$(python3 - <<'PY' 2>/dev/null || python - <<'PY'
import secrets, pathlib, os, re
root = pathlib.Path(".")
env_path = root / ".env"
text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
m = re.search(r"(?m)^MEMORY_API_SECRET=(.*)$", text)
secret = (m.group(1).strip().strip('"').strip("'") if m else "") or ""
if not secret:
    secret = secrets.token_urlsafe(32)
    if m:
        text = re.sub(r"(?m)^MEMORY_API_SECRET=.*$", f"MEMORY_API_SECRET={secret}", text)
    else:
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n# HTTP door — clients send header X-Memory-Key\n"
        text += f"MEMORY_API_SECRET={secret}\n"
        if "MEMORY_ALLOW_RELOAD=" not in text:
            text += "MEMORY_ALLOW_RELOAD=false\n"
    env_path.write_text(text, encoding="utf-8")
    print("generated")
else:
    print("exists")
home = pathlib.Path(os.path.expanduser("~")) / ".memory-agent"
home.mkdir(parents=True, exist_ok=True)
sec_file = home / "api-secret"
sec_file.write_text(secret + "\n", encoding="utf-8")
try:
    os.chmod(sec_file, 0o600)
except Exception:
    pass
print(str(sec_file))
PY
)" || true
if [ -z "${door_out:-}" ]; then
  warn "Could not ensure MEMORY_API_SECRET — set it in .env manually."
else
  status="$(printf '%s\n' "$door_out" | head -n1)"
  sec_path="$(printf '%s\n' "$door_out" | sed -n '2p')"
  if [ "$status" = "generated" ]; then info "Generated MEMORY_API_SECRET in .env"; else info "MEMORY_API_SECRET already set in .env"; fi
  [ -n "$sec_path" ] && info "Host secret file: $sec_path (Stop hook reads if env unset)"
  info "HTTP published on 127.0.0.1 only; /reload off unless MEMORY_ALLOW_RELOAD=true"
fi

# ── 3. Provider picker ──────────────────────────
echo ""
echo -e "  ${CYAN}Compose default embeddings: TEI MiniLM 384-d (no host embedder).${NC}"
echo -e "  ${CYAN}Which LLM / embedding provider do you want to use?${NC}"
echo "    1) Compose defaults (TEI MiniLM 384 + LLM from .env / host) -- recommended"
echo "    2) Local LLM override (LM Studio / Ollama) -- optional; set dims to match embed model"
echo "    3) OpenAI API"
echo "    4) Groq"
echo "    5) Other OpenAI-compatible API"
echo "    6) Skip -- I will edit .env myself"
read -r -p "  Enter 1-6 (default: 1): " choice
choice="${choice:-1}"

write_env=1
api_key="not-needed"
llm_url=""
emb_url=""
model=""
emb_model=""
emb_dims=""

case "$choice" in
  1)
    info "Keeping compose TEI MiniLM 384-d embeddings; only ensure LLM_* if present in .env"
    write_env=0
    info "docker compose overrides EMBEDDING_* to TEI (sentence-transformers/all-MiniLM-L6-v2, 384)."
    info "Edit .env LLM_BASE_URL / LLM_MODEL for your chat endpoint (host.docker.internal as needed)."
    ;;
  2)
    info "Configuring local LLM/embed override (not compose TEI default)"
    llm_url="http://host.docker.internal:1234/v1"
    emb_url="http://host.docker.internal:1234/v1"
    model="google/gemma-4-e4b"
    emb_model="text-embedding-nomic-embed-text-v1.5@q8_0"
    emb_dims="768"
    api_key="not-needed"

    read -r -p "  Local type? (1) LM Studio / (2) Ollama: " local_type
    if [ "$local_type" = "2" ]; then
      llm_url="http://host.docker.internal:11434/v1"
      emb_url="http://host.docker.internal:11434/v1"
      model="llama3.2"
      emb_model="nomic-embed-text"
      emb_dims="768"
    fi
    warn "Compose still forces TEI unless you override app.environment EMBEDDING_* in docker-compose.override.yml"
    ;;
  3)
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
  4)
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
  5)
    info "Configuring custom OpenAI-compatible API"
    read -r -p "  LLM API base URL (e.g. https://api.openai.com/v1): " llm_url
    read -r -p "  Embedding API base URL: " emb_url
    read -r -p "  API key: " api_key
    read -r -p "  LLM model: " model
    read -r -p "  Embedding model: " emb_model
    read -r -p "  Embedding dimensions: " emb_dims
    ;;
  *)
    write_env=0
    warn "Skipping auto-config — edit .env manually then re-run"
    ;;
esac

# ── 4. Write provider settings to .env ──────────
if [ "$write_env" -eq 1 ] && [ "$choice" != "6" ]; then
  # Remove old OPENAI_BASE_URL, EMBEDDING_BASE_URL, LLM_BASE_URL, LLM_API_KEY
  sed -i '/^OPENAI_BASE_URL=/d' .env
  sed -i '/^EMBEDDING_BASE_URL=/d' .env
  sed -i '/^LLM_BASE_URL=/d' .env
  sed -i '/^LLM_API_KEY=/d' .env

  # Match either legacy nomic 768 example lines or current MiniLM 384 defaults.
  if [ -n "$api_key" ] && [ "$api_key" != "not-needed" ]; then
    sed -i "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=$api_key|" .env
  fi
  sed -i "s|^EMBEDDING_MODEL=.*|EMBEDDING_MODEL=$emb_model|" .env
  sed -i "s|^EMBEDDING_DIMENSIONS=.*|EMBEDDING_DIMENSIONS=$emb_dims|" .env
  if [ -n "$model" ]; then
    sed -i "s|^LLM_MODEL=.*|LLM_MODEL=$model|" .env
  fi

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

# ── 7. Optional Claude Code plugin ─────────────
if [ "$CLAUDE" -eq 1 ]; then
  install_claude_plugin
fi

# ── 8. Harness configs ──────────────────────────
header "▸ Setup complete"
echo -e "  ${GREEN}Memory agent is running at http://localhost:8000${NC}"
echo -e "  ${GREEN}Database: PostgreSQL 16 + pgvector on port 5433${NC}"
echo -e "  ${GREEN}Auto-capture API: POST http://localhost:8000/capture${NC}"

echo ""
echo -e "  ${CYAN}── Claude Code (easiest) ──────────────────${NC}"
if [ "$CLAUDE" -eq 0 ]; then
  echo "  Re-run with --claude to install the auto-capture plugin:"
  echo "    ./scripts/setup.sh --claude-only"
else
  echo "  Plugin install attempted above. New session → turns auto-save to L0."
fi
echo ""
echo -e "  ${CYAN}── MCP Config (other tools) ───────────────${NC}"
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
echo -e "  ${CYAN}── Quick test ──────────────────────────────${NC}"
echo '  curl -s http://localhost:8000/health'
echo '  curl -s -X POST http://localhost:8000/capture -H "Content-Type: application/json" \'
echo '    -d "{\"user_id\":\"demo\",\"session_key\":\"s1\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"},{\"role\":\"assistant\",\"content\":\"hey\"}]}"'
