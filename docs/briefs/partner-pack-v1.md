# Brief: partner-pack v1

**Audience:** implementer (sonnet). Reviewer runs only after a diff exists.  
**Repo:** `d:\dev\projects\memory-agent`  
**Read first:** `.claude/rules/architecture.md` · this file · then touch code  
**Mode:** work partner only — returning senior, not day-0 intern. Not companion/brainify.

---

## Goal

A new coding session must load **who the user is**, a **thin agent craft spine**, and **relation/battle scars** without the model reinventing process from zero.

Success = partner pack available at session start (API + client path), not “hope they call get_persona.”

---

## Problem (current codebase)

| Have | Missing |
|------|---------|
| User L1 + L3 `get_persona` | Agent thin self |
| `agent_id` tag on rows (default `claude-code`) | Self/relation as first-class recall |
| Stop → `/capture` | Forced session-start **inject** of a pack |
| Search if model remembers | One bundle: other + self + relation |

`agent_id` today is scoping metadata, not “who I am.”

---

## In scope (v1 only)

1. **`GET /partner/{user_id}`** (query: optional `agent_id`, default `claude-code`)
   - Response shape (names flexible, fields required):
     - `other` — user persona summary (reuse L3 path) + top instructions
     - `self` — thin agent spine for `agent_id` (seed + stored deltas)
     - `relation` — top decisions/battles/norms (high-priority episodic or dedicated type)
     - `trust` / `stale` — same honesty as search/persona
2. **Persistence for non-user atoms**
   - Prefer extend L1 `mem_type` with `agent_self` and `relation` **or** equivalent clear split documented in architecture.
   - Seed **static** thin self from a small in-repo defaults file or config (craft norms), not from entertainment biography.
   - Runtime **deltas** append via explicit API or controlled extract path — do not dump full session into `agent_self`.
3. **Write policy**
   - User atoms: unchanged capture → worker extract.
   - `relation` / `agent_self`: written only for main shop agent (`claude-code` unless configured); no per-spawn PID identities.
   - No silent rewrite of live skills; no codescratch data in memory.
4. **Client path**
   - MCP tool `get_partner` (or document HTTP-only + plugin hook).
   - Claude plugin or `.claude/rules/memory.md`: session start loads partner pack (not only `get_persona`).
5. **Tests**
   - Partner pack empty-ish user still returns structure + seed self.
   - Instructions appear under `other`.
   - `agent_id` selects self slice.
   - Request path still does **not** call LLM (persona generate may remain on-demand as today — if used, same constraints as `get_persona`; prefer cache).
6. **Docs after behavior works**
   - Update `architecture.md` hot path + invariants.
   - Update `.claude/rules/memory.md` protocol.
   - One-line CLAUDE protocol already points here; adjust to “use get_partner” when shipped.

---

## Out of scope (do not build)

- Thick companion / entertainment persona / brainify  
- Score/decay/supersede full system (may leave ADD-only)  
- Memory → skill draft writer  
- Role-per-`flow-review` seats (v2)  
- OpenCode/Pi multi-host wiring beyond not breaking them  
- Merging codescratch  
- Rewriting extraction to be “more brain” end-to-end  

---

## Constraints (non-negotiable)

From `.claude/rules/architecture.md`:

1. No LLM / blocking embed on `/capture` `/add` hot path.  
2. DDL only via new `migrations.py` entry.  
3. `user_id` canonical lowercase.  
4. Recall degrades with `stale` — no false-empty when L1 exists.  
5. Work partner product — reject companion feature creep.  
6. Single-process pytest; extraction tests call `run_extraction` when asserting L1.

---

## Suggested touch list (not sacred)

| Area | Likely files |
|------|----------------|
| API | `src/server.py`, `src/models.py` |
| Engine | `src/memory.py` |
| Storage | `src/storage.py`, `src/migrations.py` if new types/indexes |
| MCP | `memory_mcp.py` |
| Host | `integrations/claude-code/` and/or `.claude/rules/memory.md` |
| Tests | `tests/test_memory.py` or `tests/test_partner.py` |
| Seed self | e.g. `src/partner_seed.py` or `data/agent_self_default.md` — keep tiny |

---

## Acceptance

1. `GET /partner/zaky?agent_id=claude-code` returns JSON with `other`, `self`, `relation`, trust fields.  
2. `self` non-empty even on cold DB (seed spine).  
3. After known instruction L1 exists, it appears in `other` (or clearly linked).  
4. MCP or documented host path can fetch pack at session start.  
5. `pytest tests/ -v` green single-process; no new handler-time LLM for capture.  
6. architecture.md + memory.md updated to match shipped behavior.  

---

## Implementer report back

- What you shipped (endpoints, types, inject path)  
- What you deferred  
- Test commands + results  
- Doc files updated  

Do not commit unless user explicitly asks.
