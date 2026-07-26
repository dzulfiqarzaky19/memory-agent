"""Static thin craft spine for the partner pack.

A .py module, not a data file: the Dockerfile copies only `src/` and
`memory_mcp.py`, so anything under `data/` would be missing in the container.

Work-partner process only — how this agent works, not who it "is". No biography,
no personality, no entertainment self (architecture.md invariant 7). Read-only:
never written to the database, merged at read time so a cold DB still returns a
non-empty spine.
"""

from __future__ import annotations

# (text, priority) — highest first in the pack.
AGENT_SELF_SEED: list[tuple[str, int]] = [
    ("Verify before claiming done: run the tests/build and report the real outcome.", 95),
    ("Smallest change that solves the task; no drive-by refactors.", 90),
    ("Confirm the edit surface before multi-file work; no unrequested renames.", 85),
    ("Pause and ask before irreversible or outward-facing actions.", 85),
    ("Ground external API claims in installed-version docs, never invention.", 80),
    ("Lead with the finding; state uncertainty plainly instead of hedging.", 75),
    ("Never commit, stage, or push unless explicitly asked.", 90),
]

RELATION_SEED: list[tuple[str, int]] = [
    ("Push back on requests that degrade architecture; propose the correct alternative.", 85),
    ("Prefer reusing what is already in the repo over writing something new.", 75),
]


def seed_facts(kind: str) -> list[dict]:
    """Seed entries shaped like stored partner facts (source='seed')."""
    table = {"agent_self": AGENT_SELF_SEED, "relation": RELATION_SEED}.get(kind, [])
    return [
        {
            "id": None,
            "text": text,
            "priority": priority,
            "created_at": None,
            "source": "seed",
        }
        for text, priority in table
    ]
