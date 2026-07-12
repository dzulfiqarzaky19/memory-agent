"""Wipe all data from conversations, memories, and scenarios tables."""
import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from config import DATABASE_URL

import asyncpg

async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    await conn.execute("TRUNCATE conversations, memories, scenarios")
    await conn.close()
    print("All data wiped from conversations, memories, scenarios.")

if __name__ == "__main__":
    asyncio.run(main())
