# Memory enforcement

**Preferred path:** Claude Code plugin auto-capture (Stop → `POST /capture`).
When that plugin is installed and the sidecar is up, L0 is written by the host —
do not double-write via MCP on every turn.

**user_id:** always lowercase canonical form (`zaky`, not `Zaky`). Server
normalizes; still pass lowercase from tools/hooks.

**Fallback** (plugin missing / non-Claude host / sidecar capture failing): after
every response, call `store_memories` with `user_id=zaky` and:
```
user: <their message>
assistant: <your response>
```

**Session start is host-injected.** The plugin's SessionStart hook posts the partner
pack (other + self + relation) into context automatically — do not call anything to
get it. `get_partner` (MCP) only *refreshes* it mid-session, or covers hosts without
the plugin. `get_persona` remains the generate-on-demand path when the pack shows
`(no persona cached yet)`.

**Always** use MCP for recall: `search_memories` before answering when prior context
may matter. Read the trust banner — `recall_trusted=false` / `stale` means results may
lag, and empty ≠ "user has no prefs."

Skip store only if the user says "don't save this" or the exchange is pure tool noise.
