# memory-agent — Claude Code plugin

Auto-captures each turn (Stop hook → `POST /capture`) and exposes MCP recall tools.

## Prerequisites

```bash
# from memory-agent repo root
docker compose up -d
# wait until healthy
curl -s http://localhost:8000/health
```

## Install

### One-shot (recommended)

```powershell
# Windows
.\scripts\setup.ps1 -Claude
```

```bash
# macOS / Linux
./scripts/setup.sh --claude
```

### Manual

```bash
claude plugin marketplace add /absolute/path/to/memory-agent/integrations
claude plugin install memory-agent@memory-agent-integrations
```

Or load the plugin directory directly if your Claude Code build supports local path install:

```bash
claude plugin install /absolute/path/to/memory-agent/integrations/claude-code
```

Restart Claude Code (or open a new session) after install.

## Config (optional env)

| Env | Default | Purpose |
|-----|---------|---------|
| `MEMORY_AGENT_URL` | `http://127.0.0.1:8000` | Sidecar base URL |
| `MEMORY_AGENT_USER_ID` | OS username | L0/L1 partition key |
| `MEMORY_AGENT_AGENT_ID` | `claude-code` | Optional agent scope |
| `MEMORY_API_SECRET` | (from `~/.memory-agent/api-secret`) | `X-Memory-Key` when sidecar auth is on |

## What you get

| Piece | Behavior |
|-------|----------|
| Stop hook `auto-capture.cjs` | Latest user+assistant exchange → `POST /capture` (fail-soft) |
| MCP `memory` | `search_memories`, `store_memories`, `get_persona` via docker exec |

Capture is host-driven (like Tencent OpenClaw auto-capture). You do not need the model to call `store_memories` for L0 durability.

## Uninstall

```bash
claude plugin disable memory-agent
# or
claude plugin uninstall memory-agent
```

## Logs

`~/.claude/hooks/logs/memory-auto-capture.jsonl`
