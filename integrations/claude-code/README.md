# memory-agent — Claude Code plugin

Injects the partner pack at session start (SessionStart hook → `GET /partner/{id}`),
auto-captures each turn (Stop hook → `POST /capture`), and exposes MCP recall tools.

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
| `MEMORY_AGENT_AGENT_ID` | `claude-code` | Agent scope for capture + partner pack; must match the sidecar's `PARTNER_AGENT_ID` for partner writes |
| `MEMORY_API_SECRET` | (from `~/.memory-agent/api-secret`) | `X-Memory-Key` when sidecar auth is on |

## What you get

| Piece | Behavior |
|-------|----------|
| SessionStart hook `partner-pack.cjs` | `GET /partner/{user_id}?agent_id=` → injected into context (fail-soft) |
| Stop hook `auto-capture.cjs` | Latest user+assistant exchange → `POST /capture` (fail-soft) |
| MCP `memory` | `search_memories`, `store_memories`, `get_persona`, `get_partner` via docker exec |

Both directions are host-driven. You do not need the model to call `store_memories`
for L0 durability, nor `get_persona` / `get_partner` to know who it is working with.

### Partner pack at session start

`partner-pack.cjs` runs on `SessionStart` and emits
`hookSpecificOutput.additionalContext`, so every new session opens with:

- **other** — user persona summary (cache only) + their standing instructions
- **self** — this agent's thin craft spine (seeded in-repo; non-empty on a cold DB)
- **relation** — how the two work together
- a staleness line when extraction is behind

The endpoint is compose-only — no LLM, no embedding — and the hook uses a 5s timeout
and always exits 0, so a stopped sidecar can never block or slow session start.
Failures are logged, not surfaced.

Writes (`POST /partner/{id}`) are pinned to the main shop agent
(`PARTNER_AGENT_ID`, default `claude-code`): a subagent or spawn id is rejected with
400 so per-spawn identities cannot accumulate. Reads accept any `agent_id`.

Verify by hand:

```bash
echo '{}' | node integrations/claude-code/scripts/partner-pack.cjs
# -> {"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}
```

## Uninstall

```bash
claude plugin disable memory-agent
# or
claude plugin uninstall memory-agent
```

## Logs

- `~/.claude/hooks/logs/memory-auto-capture.jsonl` — Stop / capture
- `~/.claude/hooks/logs/memory-partner-pack.jsonl` — SessionStart / partner pack
