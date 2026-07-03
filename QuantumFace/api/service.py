"""
Phase 6 — recognition service (business logic behind the API/dashboard).

Ties together: image -> align -> embed -> (encrypt) -> SQLite, and
image -> embed -> cosine match against the stored gallery -> log.

Embeddings are encrypted at rest with Kyber+AES-GCM by default (secure=True).
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image

from .. import config
from ..classical import detect, embed as _embed
from ..database import db
from ..security import crypto


class RecognitionService:
    def __init__(self, db_path=config.DB_PATH, secure=True):
        self.conn = db.connect(db_path)
        db.init_db(self.conn)
        self.secure = secure
        self.embedder, self.embedder_name = _embed.get_service_embedder()
        self.ek, self.dk = crypto.load_or_create_keys() if secure else (None, None)

    # -- image handling ---------------------------------------------------
    def _to_face(self, image_bytes: bytes) -> np.ndarray:
        pil = Image.open(io.BytesIO(image_bytes)).convert("L")
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        return detect.align_prealigned(arr)

    def _embed_bytes(self, image_bytes: bytes) -> np.ndarray:
        face = self._to_face(image_bytes)
        return self.embedder.embed(face[None, ...])[0]

    def _decrypt(self, blob: bytes) -> np.ndarray:
        return crypto.decrypt_embedding(blob, self.dk)

    # -- API operations ---------------------------------------------------
    def register(self, name: str, image_bytes: bytes) -> dict:
        vec = self._embed_bytes(image_bytes)
        if self.secure:
            blob = crypto.encrypt_embedding(vec, self.ek)
            uid = db.register_user(self.conn, name, encrypted_embedding=blob)
        else:
            uid = db.register_user(self.conn, name, embedding=vec)
        return {"id": uid, "name": name, "encrypted": self.secure,
                "embedder": self.embedder_name, "embed_dim": int(vec.shape[0])}

    def recognize(self, image_bytes: bytes, threshold: float = 0.5) -> dict:
        probe = self._embed_bytes(image_bytes)
        gallery = db.all_embeddings(self.conn, decrypt_fn=self._decrypt if self.secure else None)
        if not gallery:
            return {"matched": False, "reason": "empty gallery"}
        probe_n = probe / (np.linalg.norm(probe) + 1e-9)
        best_id, best_name, best_sim = None, None, -1.0
        for uid, name, vec in gallery:
            v = vec / (np.linalg.norm(vec) + 1e-9)
            sim = float(probe_n @ v)
            if sim > best_sim:
                best_id, best_name, best_sim = uid, name, sim
        matched = best_sim >= threshold
        db.log_recognition(self.conn, best_id if matched else None, best_sim, matched)
        return {"matched": matched, "user_id": best_id if matched else None,
                "name": best_name if matched else None,
                "similarity": best_sim, "threshold": threshold}

    def users(self):
        return db.list_users(self.conn)

    def delete(self, user_id: int) -> bool:
        return db.delete_user(self.conn, user_id)
