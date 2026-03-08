from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from pathlib import Path

import orjson


def build_request_cache_key(
    *,
    api_base: str | None,
    api_protocol: str,
    model_name: str | None,
    system_prompt: str,
    user_message: str,
    max_tokens: int | None,
) -> str:
    payload = {
        "api_base": api_base or "",
        "api_protocol": api_protocol,
        "model_name": model_name or "",
        "system_prompt": system_prompt,
        "user_message": user_message,
        "max_tokens": max_tokens,
    }
    digest = hashlib.blake2b(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS), digest_size=20)
    return digest.hexdigest()


class RequestCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._init_lock = threading.Lock()
        self._initialized = False
        self._ensure_initialized()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        return connection

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS request_cache (
                        cache_key TEXT PRIMARY KEY,
                        response_text TEXT NOT NULL,
                        created_at REAL NOT NULL
                    )
                    """
                )
            self._initialized = True

    def get(self, cache_key: str) -> str | None:
        self._ensure_initialized()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT response_text FROM request_cache WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        value = row[0]
        return value if isinstance(value, str) else None

    def set(self, cache_key: str, response_text: str) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO request_cache(cache_key, response_text, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_text = excluded.response_text,
                    created_at = excluded.created_at
                """,
                (cache_key, response_text, time.time()),
            )

    def delete(self, cache_key: str) -> None:
        self._ensure_initialized()
        with self._connect() as connection:
            connection.execute("DELETE FROM request_cache WHERE cache_key = ?", (cache_key,))
