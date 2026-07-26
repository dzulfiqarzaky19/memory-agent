"""HTTP door: X-Memory-Key + gated /reload."""
from __future__ import annotations

import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server as server_mod
from conftest import FakeEmbedding


@pytest.fixture
def locked_client(monkeypatch):
    monkeypatch.setattr(server_mod, "MEMORY_API_SECRET", "test-secret")
    monkeypatch.setattr(server_mod, "MEMORY_ALLOW_RELOAD", False)
    monkeypatch.setattr(server_mod, "create_embedding_provider", lambda: FakeEmbedding())
    with TestClient(server_mod.app) as c:
        yield c


@pytest.fixture
def reload_client(monkeypatch):
    monkeypatch.setattr(server_mod, "MEMORY_API_SECRET", "test-secret")
    monkeypatch.setattr(server_mod, "MEMORY_ALLOW_RELOAD", True)
    monkeypatch.setattr(server_mod, "create_embedding_provider", lambda: FakeEmbedding())
    with TestClient(server_mod.app) as c:
        yield c


def test_health_open_without_key(locked_client):
    r = locked_client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_search_401_without_key(locked_client):
    r = locked_client.post(
        "/search",
        json={"user_id": "test-auth", "query": "prefs"},
    )
    assert r.status_code == 401


def test_search_401_wrong_key(locked_client):
    r = locked_client.post(
        "/search",
        json={"user_id": "test-auth", "query": "prefs"},
        headers={"X-Memory-Key": "wrong"},
    )
    assert r.status_code == 401


def test_search_401_wrong_key_length(locked_client):
    # compare_digest must not 500 on length mismatch
    r = locked_client.post(
        "/search",
        json={"user_id": "test-auth", "query": "prefs"},
        headers={"X-Memory-Key": "x"},
    )
    assert r.status_code == 401


def test_search_ok_with_key(locked_client):
    r = locked_client.post(
        "/search",
        json={"user_id": "test-auth", "query": "prefs"},
        headers={"X-Memory-Key": "test-secret"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "results" in body


def test_reload_404_when_disabled(locked_client):
    r = locked_client.post(
        "/reload",
        json={"model": "x"},
        headers={"X-Memory-Key": "test-secret"},
    )
    assert r.status_code == 404


def test_reload_ok_when_enabled(reload_client):
    r = reload_client.post(
        "/reload",
        json={"model": "test-model"},
        headers={"X-Memory-Key": "test-secret"},
    )
    assert r.status_code == 200
    assert r.json()["model"] == "test-model"


def test_open_when_secret_unset(monkeypatch):
    monkeypatch.setattr(server_mod, "MEMORY_API_SECRET", "")
    monkeypatch.setattr(server_mod, "MEMORY_ALLOW_RELOAD", False)
    monkeypatch.setattr(server_mod, "create_embedding_provider", lambda: FakeEmbedding())
    with TestClient(server_mod.app) as c:
        r = c.post("/search", json={"user_id": "test-auth", "query": "prefs"})
        assert r.status_code == 200
