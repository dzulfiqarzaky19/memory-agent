from __future__ import annotations

import json
import logging
from typing import Optional

import httpx

from config import (
    EXTRACTION_MAX_MEMORIES,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_PROVIDER,
)

logger = logging.getLogger(__name__)

ALLOWED_TYPES = {"persona", "episodic", "instruction"}

SYSTEM_PROMPT = """You are a memory extraction engine. Given a conversation between a user and an assistant, extract atomic facts that would be useful for personalizing future interactions.

Extract each fact as a concise, self-contained statement with a TYPE and a PRIORITY (0-100).

TYPES:
- "persona": stable traits, preferences, identity, relationships, tools, habits (e.g. "User prefers dark mode"). Priority 60-100 (health/safety/critical identity facts 80-100).
- "episodic": objective events, decisions, or ongoing projects (e.g. "User is building a SaaS called Taskflow"). Priority 30-70.
- "instruction": an explicit long-term rule the user wants the assistant to follow (e.g. "Always answer in English", "Never use emojis", "Call me Alex"). Priority 70-100.

Rules:
1. Each fact is a single, specific piece of information.
2. Do NOT extract greetings, small talk, transient one-off requests, or the assistant's own output.
3. Do NOT repeat facts already in EXISTING MEMORIES.
4. Only extract NEW facts (ADD-only — never rephrase or update existing ones).
5. Return a JSON object with a "memories" array of objects: {"content": str, "type": str, "priority": int}.

Example output:
{"memories": [
  {"content": "User prefers dark mode in VS Code", "type": "persona", "priority": 70},
  {"content": "User is building a SaaS product called Taskflow", "type": "episodic", "priority": 55},
  {"content": "Always respond in concise bullet points", "type": "instruction", "priority": 85}
]}

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


def _normalize_atom(m) -> Optional[dict]:
    # Accept both the typed object format and a bare string (defensive).
    if isinstance(m, str):
        content = m.strip()
        return {"content": content, "type": "episodic", "priority": 50} if content else None
    if not isinstance(m, dict):
        return None
    content = str(m.get("content", "")).strip()
    if not content:
        return None
    mem_type = m.get("type", "episodic")
    if mem_type not in ALLOWED_TYPES:
        mem_type = "episodic"
    try:
        priority = int(m.get("priority", 50))
    except (TypeError, ValueError):
        priority = 50
    return {"content": content, "type": mem_type, "priority": max(0, min(100, priority))}


def _strip_markdown(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _parse_json_object(text: str) -> dict:
    """Parse first JSON object from model output (tolerates fences / trailing junk)."""
    cleaned = _strip_markdown(text)
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start < 0:
        raise json.JSONDecodeError("no JSON object in LLM response", cleaned, 0)
    parsed, _ = decoder.raw_decode(cleaned, start)
    if not isinstance(parsed, dict):
        raise json.JSONDecodeError("LLM JSON root is not an object", cleaned, start)
    return parsed


class LLMExtractor:
    def __init__(self):
        self._provider = LLM_PROVIDER
        self._model = LLM_MODEL
        self._base_url = LLM_BASE_URL.rstrip("/") if LLM_BASE_URL else ""
        self._api_key = LLM_API_KEY
        self._max_tokens = LLM_MAX_TOKENS

    def reconfigure(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                setattr(self, f"_{k}", v)

    def _require_llm(self) -> None:
        missing = [
            name
            for name, val in (
                ("LLM_BASE_URL", self._base_url),
                ("LLM_MODEL", self._model),
                ("LLM_API_KEY", self._api_key),
            )
            if not val
        ]
        if missing:
            raise RuntimeError(
                "LLM not configured — set in .env: " + ", ".join(missing)
                + (f" (LLM_PROVIDER={self._provider!r})" if self._provider else "")
            )

    async def extract_memories(
        self,
        messages_text: str,
        existing_memories: Optional[list[str]] = None,
    ) -> list[dict]:
        """Return atoms, or raise on LLM/parse failure (empty list = genuine no facts)."""
        user_prompt = build_extraction_prompt(messages_text, existing_memories)
        response_text = await self._call_llm(
            system=SYSTEM_PROMPT,
            user=user_prompt,
        )
        parsed = _parse_json_object(response_text)
        memories = parsed.get("memories", [])
        if not isinstance(memories, list):
            raise ValueError("LLM memories field is not a list")
        atoms = [atom for m in memories if (atom := _normalize_atom(m))]
        return atoms[:EXTRACTION_MAX_MEMORIES]

    async def group_into_scenarios(
        self,
        memory_texts: list[str],
        existing_scenarios: Optional[list[dict]] = None,
    ) -> list[dict]:
        memories_block = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memory_texts))

        existing_block = ""
        if existing_scenarios:
            existing_block = (
                "\nEXISTING SCENARIOS (prefer reusing these names when memories still fit):\n"
            )
            for s in existing_scenarios:
                existing_block += f"  - {s['name']}: {s['description']}\n"

        prompt = f"""Group the following memories into contextual scenarios.
A scenario is a named theme that groups related facts (e.g., "Project Taskflow", "Development Preferences", "Communication Style").

Each scenario should:
- Have a short, descriptive name (2-5 words)
- Have a 1-2 sentence description of what this theme covers
- Reference membership ONLY via memory_indices into the MEMORIES list below (1-based numbers)
- Do NOT emit memory UUIDs

If existing scenarios are provided, prefer reusing those names when memories still fit.

MEMORIES:
{memories_block}
{existing_block}

Return a JSON object: {{"scenarios": [{{"name": "...", "description": "...", "memory_indices": [1, 2, 3]}}]}}
Only include scenarios with at least 1 memory."""

        # Raise on LLM/transport/parse failure — [] only means genuine empty grouping.
        response_text = await self._call_llm(
            system="You are a memory clustering engine. Group related facts into named contextual scenarios.",
            user=prompt,
        )
        parsed = _parse_json_object(response_text)
        scenarios = parsed.get("scenarios", [])
        if not isinstance(scenarios, list):
            raise ValueError("LLM scenarios field is not a list")
        result = []
        for s in scenarios:
            if not isinstance(s, dict):
                continue
            name = s.get("name", "").strip()
            desc = s.get("description", "").strip()
            indices = s.get("memory_indices", [])
            if name and desc and isinstance(indices, list):
                # Convert 1-indexed to 0-indexed, filter valid
                valid = [
                    i - 1
                    for i in indices
                    if isinstance(i, int) and 1 <= i <= len(memory_texts)
                ]
                result.append(
                    {"name": name, "description": desc, "memory_indices": valid}
                )
        return result

    async def generate_persona(self, memories: list) -> str:
        """memories: list[str] or list[dict] with text/created_at (prefer dicts for conflict)."""
        lines: list[str] = []
        for m in memories:
            if isinstance(m, dict):
                text = (m.get("text") or "").strip()
                ts = m.get("created_at")
                stamp = ""
                if ts is not None:
                    try:
                        stamp = str(ts)[:19]
                    except Exception:
                        stamp = ""
                if text:
                    lines.append(f"- [{stamp}] {text}" if stamp else f"- {text}")
            else:
                t = str(m).strip()
                if t:
                    lines.append(f"- {t}")
        memories_text = "\n".join(lines)
        prompt = f"""Based on the following memories about a user, generate a concise persona summary.
Write it as a natural paragraph that captures who this person is, what they care about, and how they work.

Conflict rule: when two memories contradict (e.g. editor, stack, location), prefer the NEWER dated fact and omit or mark the older as historical. Do not present both as current truth.

MEMORIES (older → may be outdated; timestamps when present):
{memories_text}

Persona:"""

        # Raise on failure — caller must not cache error strings as persona.
        return await self._call_llm(
            system=(
                "You are a persona synthesis engine. Generate a concise, accurate user profile. "
                "Prefer newer memories when facts conflict."
            ),
            user=prompt,
        )

    async def _call_llm(self, system: str, user: str) -> str:
        self._require_llm()
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
            body = resp.text
            if resp.status_code >= 400:
                raise httpx.HTTPStatusError(
                    f"LLM HTTP {resp.status_code}: {body[:400]}",
                    request=resp.request,
                    response=resp,
                )
            data = _parse_json_object(body)
            try:
                content = data["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as e:
                raise RuntimeError(f"unexpected LLM response shape: {body[:400]}") from e
            if isinstance(content, list):
                # OpenAI-style multipart content blocks
                content = "".join(
                    (b.get("text") or "") if isinstance(b, dict) else str(b) for b in content
                )
            if not isinstance(content, str):
                content = str(content)
            return content
