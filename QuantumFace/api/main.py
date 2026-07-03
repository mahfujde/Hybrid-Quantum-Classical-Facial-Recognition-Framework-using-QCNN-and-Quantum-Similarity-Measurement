"""
Phase 6 — FastAPI endpoints.

    POST   /register        name + image file  -> register a user
    POST   /recognize       image file         -> best match + similarity
    GET    /users           list registered users
    DELETE /user/{id}       remove a user

Run:  uvicorn QuantumFace.api.main:app --reload

SECURITY TODO (not implemented — out of scope for this research demo):
  * authentication (API keys / OAuth) on every endpoint
  * rate limiting (e.g. slowapi) to slow brute-force probing
  * TLS termination in front of the app
Embeddings ARE encrypted at rest (Kyber+AES-GCM). See security/crypto.py threat model.
"""
from __future__ import annotations
from fastapi import FastAPI, File, Form, UploadFile, HTTPException

from .service import RecognitionService

app = FastAPI(title="QuantumFace API",
              description="RESEARCH/EDUCATION prototype — not a production biometric "
                          "system. Quantum results come from simulators only.",
              version="0.1.0")

_service: RecognitionService | None = None


def get_service() -> RecognitionService:
    global _service
    if _service is None:
        _service = RecognitionService()
    return _service


@app.get("/")
def root():
    return {"service": "QuantumFace", "status": "ok",
            "note": "research/education prototype; simulators only"}


@app.post("/register")
async def register(name: str = Form(...), file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty image")
    return get_service().register(name, data)


@app.post("/recognize")
async def recognize(file: UploadFile = File(...), threshold: float = 0.5):
    data = await file.read()
    if not data:
        raise HTTPException(400, "empty image")
    return get_service().recognize(data, threshold=threshold)


@app.get("/users")
def users():
    return {"users": get_service().users()}


@app.delete("/user/{user_id}")
def delete_user(user_id: int):
    ok = get_service().delete(user_id)
    if not ok:
        raise HTTPException(404, f"user {user_id} not found")
    return {"deleted": user_id}
