"""Content-addressed cache for model responses.

Three purposes, in order:

  Reproducibility. A coverage report must be re-derivable. If re-running the labeler
  produced different labels, a version comparison would be measuring model
  nondeterminism rather than agent quality.

  Cost. Re-running a report should not pay for the same calls twice.

  Resumability. A run stopped by the budget fuse or a network partition restarts from the
  completed calls already on disk.

The key includes the prompt version and model name. A key that ignored prompt version
would serve labels from an older prompt alongside current ones, making the report invalid.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class ResponseCache:
    """SQLite-backed, thread-safe, keyed on everything that affects the output."""

    def __init__(self, path: Path, *, namespace: str = "default",
                 seed_from: Path | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Seed from a recorded fixture on first use, so `report --llm` and `agreement` are
        # reproducible from a clean checkout without an API key. Recorded responses hold
        # only model output keyed by a hash of the request; no prompts and no transcripts.
        if seed_from is not None and not self.path.exists():
            seed = Path(seed_from)
            if seed.exists():
                import shutil
                shutil.copy2(seed, self.path)
        self.namespace = namespace
        self._lock = threading.Lock()
        # One connection with our own lock. Simpler than a pool and writes are infrequent.
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                   key TEXT PRIMARY KEY,
                   namespace TEXT NOT NULL,
                   payload TEXT NOT NULL,
                   created_at REAL NOT NULL
               )"""
        )
        self._conn.commit()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(*parts: Any) -> str:
        """Stable hash over every input that can change the output.

        sort_keys so dict ordering cannot produce a spurious miss.
        """
        blob = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE key=? AND namespace=?",
                (key, self.namespace),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: Any) -> None:
        import time
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses (key, namespace, payload, created_at) "
                "VALUES (?,?,?,?)",
                (key, self.namespace, json.dumps(value, ensure_ascii=False, default=str),
                 time.time()),
            )
            self._conn.commit()

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / total, 3) if total else 0.0}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
