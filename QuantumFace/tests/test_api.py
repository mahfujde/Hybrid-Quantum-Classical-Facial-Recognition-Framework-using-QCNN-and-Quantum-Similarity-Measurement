"""Phase 7 integration test: end-to-end register -> recognize flow via FastAPI.

Verifies the FLOW and encryption-at-rest, not embedder accuracy (the offline
SimpleEmbedder fallback is intentionally weak; FaceNet is used when available).
"""
import glob
import io

import numpy as np
import pytest
from PIL import Image

from QuantumFace import config


def _png(path):
    im = Image.open(path).convert("L")
    b = io.BytesIO(); im.save(b, format="PNG"); return b.getvalue()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from QuantumFace.api import main as api
    from QuantumFace.api.service import RecognitionService
    api._service = RecognitionService(db_path=":memory:", secure=True)
    return TestClient(api.app)


def test_register_recognize_flow(client):
    s1 = sorted(glob.glob(str(config.ORL_DIR / "s1" / "*.pgm")))
    s2 = sorted(glob.glob(str(config.ORL_DIR / "s2" / "*.pgm")))
    assert s1 and s2, "ORL dataset missing"

    r1 = client.post("/register", data={"name": "subj1"},
                     files={"file": ("a.png", _png(s1[0]), "image/png")})
    assert r1.status_code == 200
    body = r1.json()
    assert body["encrypted"] is True and body["id"] == 1

    client.post("/register", data={"name": "subj2"},
                files={"file": ("c.png", _png(s2[0]), "image/png")})

    rec = client.post("/recognize",
                      files={"file": ("b.png", _png(s1[1]), "image/png")})
    assert rec.status_code == 200
    out = rec.json()
    assert out["matched"] is True and 0.0 <= out["similarity"] <= 1.0

    assert len(client.get("/users").json()["users"]) == 2
    assert client.delete("/user/1").json() == {"deleted": 1}
    assert client.delete("/user/999").status_code == 404


def test_recognize_empty_gallery(client):
    s1 = sorted(glob.glob(str(config.ORL_DIR / "s1" / "*.pgm")))
    out = client.post("/recognize",
                      files={"file": ("b.png", _png(s1[0]), "image/png")}).json()
    assert out["matched"] is False
