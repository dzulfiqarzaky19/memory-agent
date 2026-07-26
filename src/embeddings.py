from __future__ import annotations

import abc
import asyncio
from typing import Optional

import httpx

from config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    EMBEDDING_SEND_DIMENSIONS,
    OPENAI_API_KEY,
)


def _assert_embedding_dims(vectors: list[list[float]], dims: int) -> list[list[float]]:
    for i, vec in enumerate(vectors):
        if len(vec) != dims:
            raise ValueError(
                f"embedding[{i}] dim {len(vec)} != expected dimensions={dims}"
            )
    return vectors


class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Blocking embed. Boot probes only — never call from an async request path."""
        ...

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        """Non-blocking embed. Default offloads the sync path to a thread.

        Everything on the event loop (requests and the extraction worker share it)
        must use this — a blocking HTTP call here stalls the whole sidecar.
        """
        return await asyncio.to_thread(self.embed, texts)

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        ...


class OpenAICompatibleEmbedding(EmbeddingProvider):
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        dims: int,
        send_dimensions: bool = False,
    ):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dims = dims
        self._send_dimensions = send_dimensions

    @property
    def dimensions(self) -> int:
        return self._dims

    def _payload(self, texts: list[str]) -> dict:
        payload: dict = {"input": texts, "model": self._model}
        if self._send_dimensions:
            payload["dimensions"] = self._dims
        return payload

    def _parse(self, body: dict) -> list[list[float]]:
        data = body["data"]
        vectors = [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
        return _assert_embedding_dims(vectors, self._dims)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._payload(texts),
            )
            resp.raise_for_status()
            return self._parse(resp.json())

    async def aembed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=self._payload(texts),
            )
            resp.raise_for_status()
            return self._parse(resp.json())


class LocalONNXEmbedding(EmbeddingProvider):
    def __init__(self, model_name: str, dims: int):
        self._dims = dims
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "Local embeddings require sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        vectors = [vec.tolist() for vec in embeddings]
        return _assert_embedding_dims(vectors, self._dims)


def create_embedding_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    dimensions: Optional[int] = None,
) -> EmbeddingProvider:
    provider = provider or EMBEDDING_PROVIDER
    model = model or EMBEDDING_MODEL
    dims = dimensions or EMBEDDING_DIMENSIONS

    if provider == "openai":
        return OpenAICompatibleEmbedding(
            model=model,
            api_key=OPENAI_API_KEY or "not-needed",
            base_url=EMBEDDING_BASE_URL,
            dims=dims,
            send_dimensions=EMBEDDING_SEND_DIMENSIONS,
        )
    elif provider == "local":
        return LocalONNXEmbedding(model_name=model, dims=dims)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
