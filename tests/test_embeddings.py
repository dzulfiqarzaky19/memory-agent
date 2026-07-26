from __future__ import annotations

import pytest

from embeddings import EmbeddingProvider, _assert_embedding_dims
from config import EMBEDDING_DIMENSIONS


def test_assert_embedding_dims_ok():
    vecs = [[0.1] * EMBEDDING_DIMENSIONS, [0.2] * EMBEDDING_DIMENSIONS]
    assert _assert_embedding_dims(vecs, EMBEDDING_DIMENSIONS) is vecs


def test_assert_embedding_dims_mismatch():
    with pytest.raises(ValueError, match=r"embedding\[0\] dim 3 !="):
        _assert_embedding_dims([[0.1] * 3], EMBEDDING_DIMENSIONS)


class _WrongWidthEmbedding(EmbeddingProvider):
    """Mimics a provider that lies about dimensions property vs live vectors."""

    def __init__(self, claimed: int, actual: int):
        self._claimed = claimed
        self._actual = actual

    @property
    def dimensions(self) -> int:
        return self._claimed

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = [[0.1] * self._actual for _ in texts]
        return _assert_embedding_dims(vectors, self._claimed)


def test_embed_path_rejects_live_width_mismatch():
    bad = _WrongWidthEmbedding(claimed=EMBEDDING_DIMENSIONS, actual=3)
    with pytest.raises(ValueError, match=r"embedding\[0\] dim 3 !="):
        bad.embed(["dim-check"])
