"""Hash-based JSON cache for clip highlight scores."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


_CACHE_DIR = Path.home() / ".cache" / "fps-mv" / "scores"


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def make_key(video_path: Path, params: dict) -> str:
    mtime = str(video_path.stat().st_mtime) if video_path.exists() else "0"
    raw = str(video_path.resolve()) + mtime + json.dumps(params, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def cache_get(key: str) -> list[dict] | None:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def cache_put(key: str, segments: list[dict]) -> None:
    _cache_path(key).write_text(json.dumps(segments))
