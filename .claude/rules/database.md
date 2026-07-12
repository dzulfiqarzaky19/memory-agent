---
paths:
  - "src/storage.py"
  - "tests/**"
  - "reset_db.py"
---

# Database & migrations (memory-agent)

- When pytest (or app startup) hangs with NO output against the pgvector DB, a startup DDL migration is lock-blocked: `Storage.initialize()` runs `SCHEMA_SQL` on every boot, and `ALTER TABLE memories ADD COLUMN … / CREATE INDEX` needs `ACCESS EXCLUSIVE`, which stalls behind any `idle in transaction` connection or the running `app` container. Diagnose with `docker exec memory-agent-db-1 psql -U postgres -d memory_agent -c "SELECT pid,state,wait_event_type,left(query,60) FROM pg_stat_activity WHERE datname='memory_agent' AND pid<>pg_backend_pid();"`; clear ALL blockers with `SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='memory_agent' AND pid<>pg_backend_pid();` — `docker compose stop app` alone is NOT enough, since the IDE's DB explorer also leaves `idle in transaction` connections holding `ACCESS SHARE` on `memories`/`scenarios`. Confirm a fast migration by applying the ALTER manually with `SET lock_timeout='8s';` — a `canceling statement due to lock timeout` proves a lock holder remains.
- Run the pytest suite in a SINGLE process — concurrent runs against the shared DB deadlock on the fixture `TRUNCATE` and cross-contaminate fixed user_ids (e.g. `test-dedup`), giving false `count==2`/`assert None` failures. Kill strays with `Get-Process python | Stop-Process -Force` and verify `pg_stat_activity` shows 0 backends before rerunning.
- The `memories` table's type column is `mem_type`, not `type` (`type` is a reserved/absent name — `SELECT type FROM memories` errors `column "type" does not exist`) — use `SELECT * FROM memories WHERE user_id=...` or name `mem_type` explicitly.
