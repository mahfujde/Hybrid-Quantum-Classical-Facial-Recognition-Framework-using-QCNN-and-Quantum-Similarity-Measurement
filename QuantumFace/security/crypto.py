"""
Phase 6 — post-quantum encryption of stored face embeddings.

THREAT MODEL (read carefully — the scope is narrow and specific):

  PROTECTS AGAINST : offline theft of the embedding database. An attacker who steals
                     `quantumface.db` sees only Kyber-wrapped AES-GCM ciphertext and
                     cannot recover the raw embeddings without the Kyber decapsulation
                     (secret) key. "Post-quantum" = the key-encapsulation (ML-KEM /
                     Kyber, FIPS 203) resists an attacker with a quantum computer.

  DOES NOT PROTECT : model-inversion / template-reconstruction attacks, presentation
                     (spoofing) attacks against the camera, a compromised endpoint
                     that holds the decapsulation key in memory, or traffic in flight
                     (use TLS for that). Encrypting embeddings at rest is ONE control,
                     not "unhackable identity".

Construction: KEM-DEM envelope.
  * Kyber ML-KEM-512 encapsulates a fresh 32-byte shared secret per embedding.
  * That secret keys AES-256-GCM, which encrypts the embedding bytes (authenticated).
  * Stored blob = kem_ciphertext ‖ nonce ‖ aesgcm_ciphertext(+tag).
The decapsulation (secret) key is the only thing that must stay secret.
"""
from __future__ import annotations
import os
import struct
import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from kyber_py.ml_kem import ML_KEM_512

from .. import config

_EK_PATH = config.KEY_DIR / "kyber_ek.bin"   # encapsulation (public) key
_DK_PATH = config.KEY_DIR / "kyber_dk.bin"   # decapsulation (secret) key — keep safe!


def generate_keypair(persist: bool = True):
    ek, dk = ML_KEM_512.keygen()
    if persist:
        _EK_PATH.write_bytes(ek)
        _DK_PATH.write_bytes(dk)
        os.chmod(_DK_PATH, 0o600)
    return ek, dk


def load_or_create_keys():
    if _EK_PATH.exists() and _DK_PATH.exists():
        return _EK_PATH.read_bytes(), _DK_PATH.read_bytes()
    return generate_keypair(persist=True)


def encrypt_embedding(embedding: np.ndarray, ek: bytes) -> bytes:
    """Kyber-encapsulate a fresh AES key, then AES-256-GCM the embedding bytes."""
    emb = np.asarray(embedding, dtype=np.float32)
    payload = struct.pack("<I", emb.shape[0]) + emb.tobytes()
    shared_key, kem_ct = ML_KEM_512.encaps(ek)          # shared_key: 32 bytes
    aes = AESGCM(shared_key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, payload, associated_data=b"quantumface-embedding")
    return struct.pack("<H", len(kem_ct)) + kem_ct + nonce + ct


def decrypt_embedding(blob: bytes, dk: bytes) -> np.ndarray:
    (kem_len,) = struct.unpack("<H", blob[:2])
    kem_ct = blob[2:2 + kem_len]
    nonce = blob[2 + kem_len:2 + kem_len + 12]
    ct = blob[2 + kem_len + 12:]
    shared_key = ML_KEM_512.decaps(dk, kem_ct)
    aes = AESGCM(shared_key)
    payload = aes.decrypt(nonce, ct, associated_data=b"quantumface-embedding")
    (n,) = struct.unpack("<I", payload[:4])
    return np.frombuffer(payload[4:4 + n * 4], dtype=np.float32).copy()


if __name__ == "__main__":
    ek, dk = generate_keypair(persist=False)
    v = np.random.randn(512).astype(np.float32)
    blob = encrypt_embedding(v, ek)
    back = decrypt_embedding(blob, dk)
    print(f"variant={config.KYBER_VARIANT} blob={len(blob)}B roundtrip_ok={np.allclose(v, back)}")
