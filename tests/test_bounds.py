"""Payload bounds + bad-input status codes (Chunk D)."""

from __future__ import annotations

import pytest

from models import MAX_CONTENT_CHARS, MAX_MESSAGES


def test_oversized_content_rejected(client):
    r = client.post(
        "/add",
        json={
            "user_id": "test-bounds",
            "messages": [{"role": "user", "content": "x" * (MAX_CONTENT_CHARS + 1)}],
        },
    )
    assert r.status_code == 422


def test_too_many_messages_rejected(client):
    r = client.post(
        "/capture",
        json={
            "user_id": "test-bounds",
            "session_key": "test-bounds-sess",
            "messages": [
                {"role": "user", "content": "hi"} for _ in range(MAX_MESSAGES + 1)
            ],
        },
    )
    assert r.status_code == 422


def test_oversized_user_id_rejected(client):
    r = client.post(
        "/search",
        json={"user_id": "t" * 201, "query": "hi"},
    )
    assert r.status_code == 422


def test_hook_sized_payload_accepted(client):
    """Server caps must sit above the hook's 2000/2500 clip — never reject real capture."""
    r = client.post(
        "/add",
        json={
            "user_id": "test-bounds-ok",
            "messages": [
                {"role": "user", "content": "u" * 2000},
                {"role": "assistant", "content": "a" * 2500},
            ],
        },
    )
    assert r.status_code == 200


@pytest.mark.parametrize("path", ["/persona/%20", "/scenarios/%20"])
def test_whitespace_user_id_path_route_is_400(client, path):
    """ids.canonicalize_user_id raises ValueError — must surface as 400, not 500."""
    r = client.get(path)
    assert r.status_code == 400
    assert "user_id" in r.json()["detail"]


def test_whitespace_user_id_body_route_is_400(client):
    r = client.post("/search", json={"user_id": " ", "query": "hi"})
    assert r.status_code == 400
