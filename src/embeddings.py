from __future__ import annotations

import abc
from typing import Optional

import httpx

from config import (
    EMBEDDING_BASE_URL,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)


class EmbeddingProvider(abc.ABC):
    @abc.abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @property
    @abc.abstractmethod
    def dimensions(self) -> int:
        ...


class OpenAICompatibleEmbedding(EmbeddingProvider):
    def __init__(self, model: str, api_key: str, base_url: str, dims: int):
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._dims = dims

    @property
    def dimensions(self) -> int:
        return self._dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"input": texts, "model": self._model, "dimensions": self._dims},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]


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
        return [vec.tolist() for vec in embeddings]


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
            api_key=OPENAI_API_KEY,
            base_url=EMBEDDING_BASE_URL,
            dims=dims,
        )
    elif provider == "local":
        return LocalONNXEmbedding(model_name=model, dims=dims)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
