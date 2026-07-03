"""
Phase 6 — SQLite storage.

Two tables:
  users            (id, name, embedding, encrypted_embedding, date)
  recognition_log  (id, user_id, similarity, matched, timestamp)

`embedding` is stored ONLY when encryption is disabled (dev convenience). In the
default secure mode we store `encrypted_embedding` (Kyber+AES-GCM blob) and leave the
plaintext column NULL. See security/crypto.py for the threat model.
"""
from __future__ import annotations
import datetime as _dt
import sqlite3
from pathlib import Path

import numpy as np

from .. import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    embedding           BLOB,            -- plaintext (dev only; NULL in secure mode)
    encrypted_embedding BLOB,            -- Kyber+AES-GCM blob (secure mode)
    date                TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS recognition_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER,
    similarity REAL,
    matched    INTEGER,
    timestamp  TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""


def connect(db_path: Path | str = config.DB_PATH) -> sqlite3.Connection:
    # check_same_thread=False: FastAPI runs sync endpoints in a threadpool; this demo
    # serialises access so cross-thread use of one connection is safe here.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection):
    conn.executescript(_SCHEMA)
    conn.commit()


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def register_user(conn, name: str, *, embedding: np.ndarray | None = None,
                  encrypted_embedding: bytes | None = None) -> int:
    emb_blob = (np.asarray(embedding, dtype=np.float32).tobytes()
                if embedding is not None else None)
    cur = conn.execute(
        "INSERT INTO users (name, embedding, encrypted_embedding, date) VALUES (?,?,?,?)",
        (name, emb_blob, encrypted_embedding, _now()))
    conn.commit()
    return int(cur.lastrowid)


def list_users(conn) -> list[dict]:
    rows = conn.execute(
        "SELECT id, name, date, (encrypted_embedding IS NOT NULL) AS encrypted "
        "FROM users ORDER BY id").fetchall()
    return [dict(r) for r in rows]


def get_user(conn, user_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    return dict(r) if r else None


def all_embeddings(conn, decrypt_fn=None) -> list[tuple[int, str, np.ndarray]]:
    """Return [(id, name, embedding_vector)] decrypting if a decrypt_fn is given."""
    out = []
    for r in conn.execute("SELECT id, name, embedding, encrypted_embedding FROM users"):
        if r["encrypted_embedding"] is not None and decrypt_fn is not None:
            vec = decrypt_fn(r["encrypted_embedding"])
        elif r["embedding"] is not None:
            vec = np.frombuffer(r["embedding"], dtype=np.float32)
        else:
            continue
        out.append((r["id"], r["name"], np.asarray(vec, dtype=np.float32)))
    return out


def delete_user(conn, user_id: int) -> bool:
    conn.execute("DELETE FROM recognition_log WHERE user_id=?", (user_id,))
    cur = conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    return cur.rowcount > 0


def log_recognition(conn, user_id: int | None, similarity: float, matched: bool):
    conn.execute(
        "INSERT INTO recognition_log (user_id, similarity, matched, timestamp) "
        "VALUES (?,?,?,?)", (user_id, float(similarity), int(matched), _now()))
    conn.commit()


def recent_logs(conn, limit=20) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM recognition_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    conn = connect(":memory:")
    init_db(conn)
    uid = register_user(conn, "alice", embedding=np.ones(8, dtype=np.float32))
    log_recognition(conn, uid, 0.9, True)
    print("users:", list_users(conn))
    print("embeddings:", [(i, n, v.shape) for i, n, v in all_embeddings(conn)])
    print("logs:", recent_logs(conn))
