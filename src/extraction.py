from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MAX_TOKENS, LLM_MODEL, LLM_PROVIDER

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a memory extraction engine. Given a conversation between a user and an assistant, extract atomic facts that would be useful for personalizing future interactions.

Rules:
1. Extract facts as concise, self-contained statements
2. Each fact should be a single, specific piece of information
3. Include preferences, habits, projects, relationships, tools, and any personal details
4. Do NOT extract greetings, generic phrases, or transient information
5. Do NOT update or modify existing facts — only extract NEW facts
6. Return a JSON object with a "memories" array of strings

Example output:
{"memories": ["User prefers dark mode in VS Code", "User is building a SaaS product called Taskflow", "User uses PostgreSQL for all projects"]}

If no extractable facts, return: {"memories": []}"""


def build_extraction_prompt(
    new_messages: str,
    existing_memories: Optional[list[str]] = None,
) -> str:
    parts = []
    if existing_memories:
        parts.append("EXISTING MEMORIES (for dedup — do NOT repeat these):\n")
        for i, m in enumerate(existing_memories, 1):
            parts.append(f"  {i}. {m}")
        parts.append("")

    parts.append("NEW CONVERSATION:\n")
    parts.append(new_messages)
    parts.append("\n\nExtract new atomic facts from the conversation above.")
    return "\n".join(parts)


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


class LLMExtractor:
    def __init__(self):
        self._provider = LLM_PROVIDER
        self._model = LLM_MODEL
        self._base_url = (LLM_BASE_URL or "https://api.openai.com/v1").rstrip("/")
        self._api_key = LLM_API_KEY
        self._max_tokens = LLM_MAX_TOKENS

    async def extract_memories(
        self,
        messages_text: str,
        existing_memories: Optional[list[str]] = None,
    ) -> list[str]:
        user_prompt = build_extraction_prompt(messages_text, existing_memories)

        try:
            response_text = await self._call_llm(
                system=SYSTEM_PROMPT,
                user=user_prompt,
            )
            response_text = _strip_markdown(response_text)
            parsed = json.loads(response_text)
            memories = parsed.get("memories", [])
            if not isinstance(memories, list):
                return []
            return [m.strip() for m in memories if isinstance(m, str) and m.strip()]
        except Exception as e:
            logger.error(f"Memory extraction failed: {e}")
            return []

    async def group_into_scenarios(
        self,
        memory_texts: list[str],
        existing_scenarios: Optional[list[dict]] = None,
    ) -> list[dict]:
        memories_block = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memory_texts))

        existing_block = ""
        if existing_scenarios:
            existing_block = "\nEXISTING SCENARIOS (update these if new memories fit, or create new ones):\n"
            for s in existing_scenarios:
                existing_block += f"  - {s['name']}: {s['description']} (IDs: {', '.join(s.get('memory_ids', []))})\n"

        prompt = f"""Group the following memories into contextual scenarios.
A scenario is a named theme that groups related facts (e.g., "Project Taskflow", "Development Preferences", "Communication Style").

Each scenario should:
- Have a short, descriptive name (2-5 words)
- Have a 1-2 sentence description of what this theme covers
- Reference the memory numbers that belong to it

If existing scenarios are provided, prefer adding new memories to existing scenarios when they fit.

MEMORIES:
{memories_block}
{existing_block}

Return a JSON object: {{"scenarios": [{{"name": "...", "description": "...", "memory_indices": [1, 2, 3]}}]}}
Only include scenarios with at least 1 memory."""

        try:
            response_text = await self._call_llm(
                system="You are a memory clustering engine. Group related facts into named contextual scenarios.",
                user=prompt,
            )
            response_text = _strip_markdown(response_text)
            parsed = json.loads(response_text)
            scenarios = parsed.get("scenarios", [])
            if not isinstance(scenarios, list):
                return []
            result = []
            for s in scenarios:
                if not isinstance(s, dict):
                    continue
                name = s.get("name", "").strip()
                desc = s.get("description", "").strip()
                indices = s.get("memory_indices", [])
                if name and desc and isinstance(indices, list):
                    # Convert 1-indexed to 0-indexed, filter valid
                    valid = [i - 1 for i in indices if isinstance(i, int) and 1 <= i <= len(memory_texts)]
                    result.append({"name": name, "description": desc, "memory_indices": valid})
            return result
        except Exception as e:
            logger.error(f"Scenario grouping failed: {e}")
            return []

    async def generate_persona(self, memories: list[str]) -> str:
        memories_text = "\n".join(f"- {m}" for m in memories)
        prompt = f"""Based on the following memories about a user, generate a concise persona summary.
Write it as a natural paragraph that captures who this person is, what they care about, and how they work.

MEMORIES:
{memories_text}

Persona:"""

        try:
            return await self._call_llm(
                system="You are a persona synthesis engine. Generate a concise, accurate user profile from extracted memories.",
                user=prompt,
            )
        except Exception as e:
            logger.error(f"Persona generation failed: {e}")
            return "No persona available yet."

    async def _call_llm(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=300) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "max_tokens": self._max_tokens,
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
