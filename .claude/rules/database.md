---
paths:
  - "src/storage.py"
  - "tests/**"
  - "reset_db.py"
---

# Database & migrations (memory-agent)

- Schema changes go in `src/migrations.py` as a NEW numbered entry in `MIGRATIONS` — never edit an applied migration, and never add DDL to `Storage.initialize()`. A steady-state boot logs `migrations applied=none` and issues zero DDL; if a boot suddenly applies something you didn't add, an existing migration was mutated.
- When pytest (or app startup) hangs with NO output against the pgvector DB, a startup DDL migration is lock-blocked: a pending migration's `ALTER TABLE memories ADD COLUMN … / CREATE INDEX` needs `ACCESS EXCLUSIVE`, which stalls behind any `idle in transaction` connection or the running `app` container. Diagnose with `docker exec memory-agent-db-1 psql -U postgres -d memory_agent -c "SELECT pid,state,wait_event_type,left(query,60) FROM pg_stat_activity WHERE datname='memory_agent' AND pid<>pg_backend_pid();"`; clear ALL blockers with `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='memory_agent' AND pid<>pg_backend_pid();` — `docker compose stop app` alone is NOT enough, since the IDE's DB explorer also leaves `idle in transaction` connections holding `ACCESS SHARE` on `memories`/`scenarios`. Confirm a fast migration by applying the ALTER manually with `SET lock_timeout='8s';` — a `canceling statement due to lock timeout` proves a lock holder remains.
- Run the pytest suite in a SINGLE process — concurrent runs against the shared DB deadlock on the fixture `TRUNCATE` and cross-contaminate fixed user_ids (e.g. `test-dedup`), giving false `count==2`/`assert None` failures. Kill strays with `Get-Process python | Stop-Process -Force` and verify `pg_stat_activity` shows 0 backends before rerunning.
- The `memories` table's type column is `mem_type`, not `type` (`type` is a reserved/absent name — `SELECT type FROM memories` errors `column "type" does not exist`) — use `SELECT * FROM memories WHERE user_id=...` or name `mem_type` explicitly.
- Never call the LLM or a blocking embed from a request handler. Writes enqueue an `extraction_jobs` row and return (~5ms); `src/worker.py` drains it. On the event loop use `embedder.aembed()` — sync `embed()` is for the boot probe only, and blocking there stalls every concurrent request, not just the caller.
- Tests that assert on extraction must call `engine.run_extraction(uid)` explicitly — `add()`/`capture()` only queue now, and `run_extraction` RAISES on failure so the queue can retry (`extract_status` no longer reports `"failed"`).
