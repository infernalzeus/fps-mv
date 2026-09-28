"""Tests for score cache."""
import json
import pytest
from pathlib import Path
from unittest.mock import patch

from src.utils.cache import make_key, cache_get, cache_put


@pytest.fixture(autouse=True)
def tmp_cache(tmp_path, monkeypatch):
    """Redirect cache to a temp directory for isolation."""
    import src.utils.cache as cache_mod
    monkeypatch.setattr(cache_mod, "_CACHE_DIR", tmp_path / "scores")


def _fake_path(tmp_path: Path) -> Path:
    p = tmp_path / "clip.mp4"
    p.write_bytes(b"fake")
    return p


def test_cache_miss(tmp_path):
    key = make_key(_fake_path(tmp_path), {"threshold": 0.5})
    assert cache_get(key) is None


def test_cache_roundtrip(tmp_path):
    p = _fake_path(tmp_path)
    key = make_key(p, {"threshold": 0.5})
    segs = [{"start": 1.0, "end": 4.0, "score": 0.8, "kill_time": None}]
    cache_put(key, segs)
    result = cache_get(key)
    assert result == segs


def test_cache_key_changes_with_params(tmp_path):
    p = _fake_path(tmp_path)
    key1 = make_key(p, {"threshold": 0.5})
    key2 = make_key(p, {"threshold": 0.6})
    assert key1 != key2


def test_cache_overwrite(tmp_path):
    p = _fake_path(tmp_path)
    key = make_key(p, {"threshold": 0.5})
    cache_put(key, [{"start": 0.0, "end": 3.0, "score": 0.7, "kill_time": None}])
    new_segs = [{"start": 5.0, "end": 8.0, "score": 0.9, "kill_time": 6.5}]
    cache_put(key, new_segs)
    assert cache_get(key) == new_segs
