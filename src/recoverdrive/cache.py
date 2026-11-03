"""Content-addressed JSON cache (SQLite). Makes SAM 3 and VLM results reusable across methods, seeds and reruns."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np


def digest(obj: Any) -> str:
    """Stable sha256 of a JSON-serializable object (dict key order does not matter)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def image_digest(image: np.ndarray) -> str:
    h = hashlib.sha1(str(image.shape).encode())
    h.update(np.ascontiguousarray(image).tobytes())
    return h.hexdigest()


class KVCache:
    """Thread-safe key -> JSON value store backed by one SQLite file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False, timeout=120)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT NOT NULL)")
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k = ?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key: str, value: Any) -> None:
        text = json.dumps(value)
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv (k, v) VALUES (?, ?)", (key, text))
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
