"""Phase 6 tests: Kyber+AES-GCM roundtrip, tamper-detection, SQLite DAO."""
import numpy as np
import pytest

from QuantumFace.security import crypto
from QuantumFace.database import db


def test_crypto_roundtrip():
    ek, dk = crypto.generate_keypair(persist=False)
    v = np.random.randn(256).astype(np.float32)
    blob = crypto.encrypt_embedding(v, ek)
    back = crypto.decrypt_embedding(blob, dk)
    assert np.allclose(v, back, atol=1e-6)
    assert len(blob) > 256                      # includes KEM ciphertext + nonce + tag


def test_crypto_wrong_key_fails():
    ek1, _ = crypto.generate_keypair(persist=False)
    _, dk2 = crypto.generate_keypair(persist=False)
    blob = crypto.encrypt_embedding(np.ones(8, dtype=np.float32), ek1)
    with pytest.raises(Exception):
        crypto.decrypt_embedding(blob, dk2)     # AES-GCM auth fails


def test_db_register_list_delete():
    conn = db.connect(":memory:")
    db.init_db(conn)
    uid = db.register_user(conn, "carol", embedding=np.arange(8, dtype=np.float32))
    assert db.list_users(conn)[0]["name"] == "carol"
    embs = db.all_embeddings(conn)
    assert embs[0][2].shape == (8,)
    db.log_recognition(conn, uid, 0.8, True)
    assert db.recent_logs(conn)[0]["matched"] == 1
    assert db.delete_user(conn, uid) is True
    assert db.list_users(conn) == []


def test_db_stores_encrypted_only_in_secure_mode():
    conn = db.connect(":memory:")
    db.init_db(conn)
    ek, dk = crypto.generate_keypair(persist=False)
    blob = crypto.encrypt_embedding(np.ones(16, dtype=np.float32), ek)
    db.register_user(conn, "dave", encrypted_embedding=blob)
    row = conn.execute("SELECT embedding, encrypted_embedding FROM users").fetchone()
    assert row["embedding"] is None and row["encrypted_embedding"] is not None
    got = db.all_embeddings(conn, decrypt_fn=lambda b: crypto.decrypt_embedding(b, dk))
    assert np.allclose(got[0][2], np.ones(16))
