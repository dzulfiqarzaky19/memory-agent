# Code structure (sibling MCP)

`codescratch` is a **separate** local graph tool (not part of this Python package).
No shared DB/process with memory-agent.

## When to use which

| Need | Tool |
|------|------|
| User prefs / past decisions / persona | `search_memories` / `get_persona` |
| Where is X defined, who calls it, blast radius | `cs_search` / `cs_explore` / `cs_callers` / `cs_impact` |
| Graph stale after edits | `cs_reindex` (or trust banner `reindex:` cmd) |

## Rules

- Read `trust:` on every `cs_*` reply. `stale` → reindex. `partial` / `conf=weak` / unresolved → do not treat absence as proof.
- Critical paths still require reading source.
- Index lives at `<repo>/.codescratch/` — gitignored if present; rebuild with `cs_reindex` or CLI.