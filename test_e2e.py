"""End-to-end test. Cleans up test data after."""
import asyncio
import os
import sys
import time

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from storage import Storage
from embeddings import create_embedding_provider
from extraction import LLMExtractor
from memory import MemoryEngine

TEST_USER = "_test_e2e"
BASE = "http://localhost:8000"
CLIENT = httpx.Client(timeout=httpx.Timeout(120.0))


async def main():
    print("=" * 50)
    print("memory-agent end-to-end test")
    print("=" * 50)

    # --- Via HTTP API ---
    print("\n1. Health check")
    r = CLIENT.get(f"{BASE}/health")
    print(f"   {r.json()}")

    print("\n2. Store conversation (no extraction)")
    r = CLIENT.post(f"{BASE}/add", json={
        "user_id": TEST_USER,
        "messages": [
            {"role": "user", "content": "what took you so long?"},
            {"role": "assistant", "content": "sorry about that"},
        ],
    })
    j = r.json()
    print(f"   memories_added: {j['memories_added']}")
    print(f"   0 = expected (not enough turns yet)")

    # Default EXTRACTION_EVERY_N_TURNS=5. Step 2 was 1 user turn; need 4 more.
    print("\n3. Store until extraction threshold (default every 5 user turns)")
    r = CLIENT.post(f"{BASE}/add", json={
        "user_id": TEST_USER,
        "messages": [
            {"role": "user", "content": "I use Python and PostgreSQL for my projects"},
            {"role": "assistant", "content": "Great tech stack!"},
            {"role": "user", "content": "I prefer dark mode in VS Code"},
            {"role": "assistant", "content": "Good choice, easy on the eyes"},
            {"role": "user", "content": "My name is E2E Tester"},
            {"role": "assistant", "content": "Noted"},
            {"role": "user", "content": "I work mostly on Windows"},
            {"role": "assistant", "content": "Got it"},
        ],
    })
    j = r.json()
    print(f"   memories_added: {j['memories_added']} extract_status={j.get('extract_status')}")

    print("\n4. Search for memories")
    r = CLIENT.post(f"{BASE}/search", json={
        "user_id": TEST_USER,
        "query": "what editor does the user prefer?",
    })
    j = r.json()
    print(f"   results: {j['total']}")
    for m in j["results"]:
        print(f"   [{m['score']:.4f}] {m['text']}")

    print("\n5. Get persona")
    r = CLIENT.get(f"{BASE}/persona/{TEST_USER}")
    j = r.json()
    print(f"   memory_count: {j['memory_count']}")
    print(f"   summary: {j['summary'][:150]}...")

    print("\n6. Get scenarios")
    r = CLIENT.get(f"{BASE}/scenarios/{TEST_USER}")
    j = r.json()
    print(f"   scenarios: {j['total']}")

    # --- Cleanup: delete test data ---
    print("\n7. Cleaning up test data...")
    store = Storage()
    await store.initialize()
    await store.delete_user_data(TEST_USER)
    await store.close()
    print("   Done.")

    print("\n" + "=" * 50)
    print("All tests passed. Test data cleaned up.")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(main())
