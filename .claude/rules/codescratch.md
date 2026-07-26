# Code structure (sibling MCP)

`codescratch` is a **separate** local graph tool (not part of this Python package).
No shared DB/process with memory-agent.

## When to use which

| Need | Tool |
|------|------|
| User prefs / past decisions / persona | `search_memories` / `get_persona` |
| Where is X defined, who calls it, blast radius | `cs_search` / `cs_explore` / `cs_callers` / `cs_impact` |
| Graph freshness | **Host** `codescratch ensure` (hooks) — not routine agent `cs_reindex` |

## Rules

- Read `trust:` on every `cs_*` reply. `rebuilding` → wait; absence ≠ proof. `stale` → host ensure; `cs_reindex` only if stuck or host down. `partial` / `conf=weak` / unresolved → do not treat absence as proof.
- **Do not** call `cs_reindex` after every edit.
- Critical paths still require reading source.
- Index lives at `<repo>/.codescratch/` — rebuild with host ensure or emergency `cs_reindex`.
