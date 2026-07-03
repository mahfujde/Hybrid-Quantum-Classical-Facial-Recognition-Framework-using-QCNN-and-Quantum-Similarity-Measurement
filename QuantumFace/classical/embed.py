"""
Face embedding for the classical baseline (Phase 1).

Primary path: pretrained **FaceNet** (InceptionResnetV1, VGGFace2 weights) via
facenet-pytorch -> 512-dim embeddings. We do NOT train FaceNet from scratch.

Offline fallback: a classical **eigenface** (PCA-on-pixels) embedder. This exists so
the pipeline still produces *real measured numbers* on machines/CI where the FaceNet
pretrained weights cannot be downloaded (they are hosted on a GitHub release CDN that
some networks block). The embedding source used is always recorded in results, so no
number is ever ambiguous about where it came from.
"""
from __future__ import annotations
import os
import warnings
import numpy as np

from .. import config

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_facenet_batch(images: np.ndarray):
    """images: (N,H,W) float32 in [0,1] grayscale -> torch (N,3,160,160) standardized."""
    import torch
    from PIL import Image
    out = np.empty((len(images), config.FACE_SIZE, config.FACE_SIZE), dtype=np.float32)
    for i, im in enumerate(images):
        pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((config.FACE_SIZE, config.FACE_SIZE), Image.BILINEAR)
        out[i] = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(out).unsqueeze(1).repeat(1, 3, 1, 1)  # (N,3,160,160)
    x = (x * 255.0 - 127.5) / 128.0                             # FaceNet standardization
    return x


# ---------------------------------------------------------------------------
# FaceNet
# ---------------------------------------------------------------------------
class FaceNetEmbedder:
    name = "facenet-vggface2-512d"
    dim = config.EMBED_DIM

    def __init__(self):
        import torch
        from facenet_pytorch import InceptionResnetV1
        self.torch = torch
        # This line downloads pretrained weights on first use; may raise on
        # networks that block the GitHub release-assets CDN.
        self.model = InceptionResnetV1(pretrained=config.FACENET_PRETRAINED).eval()

    def embed(self, images: np.ndarray) -> np.ndarray:
        x = _to_facenet_batch(images)
        embs = []
        with self.torch.no_grad():
            for i in range(0, len(x), 32):
                embs.append(self.model(x[i:i + 32]).cpu().numpy())
        return np.concatenate(embs, 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Eigenface (offline fallback)
# ---------------------------------------------------------------------------
class EigenfaceEmbedder:
    """Classical PCA-on-pixels embedding — the textbook 'eigenface' baseline.

    Must be .fit() on training images before .embed(). Deterministic (SEED).
    """
    name = "eigenface-pca-pixels"

    def __init__(self, dim: int = 128):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        self.dim = dim
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=dim, whiten=True, random_state=config.SEED)
        self._fitted = False

    def fit(self, images: np.ndarray):
        flat = images.reshape(len(images), -1)
        self.dim = min(self.dim, min(flat.shape) - 1)
        self._pca.n_components = self.dim
        z = self._scaler.fit_transform(flat)
        self._pca.fit(z)
        self._fitted = True
        return self

    def embed(self, images: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("EigenfaceEmbedder.embed() before .fit()")
        flat = images.reshape(len(images), -1)
        return self._pca.transform(self._scaler.transform(flat)).astype(np.float32)


# ---------------------------------------------------------------------------
# Simple (fit-free) embedder — offline API/demo fallback
# ---------------------------------------------------------------------------
class SimpleEmbedder:
    """Downsample -> flatten -> L2-normalise. No pretrained weights, no fit.

    NOT a good recogniser — it exists so the API/demo and its integration test run
    end-to-end fully offline. Real deployments use FaceNetEmbedder.
    """
    name = "simple-downsample-l2"

    def __init__(self, side: int = 16):
        self.side = side
        self.dim = side * side

    def embed(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image
        out = np.empty((len(images), self.dim), dtype=np.float32)
        for i, im in enumerate(images):
            pil = Image.fromarray((np.clip(im, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((self.side, self.side), Image.BILINEAR)
            v = np.asarray(pil, dtype=np.float32).ravel()
            out[i] = v / (np.linalg.norm(v) + 1e-9)
        return out


def _force_offline() -> bool:
    """CI/Docker determinism: set QF_FORCE_OFFLINE_EMBEDDER=1 to skip the FaceNet weight
    download and always use the offline embedders."""
    return os.environ.get("QF_FORCE_OFFLINE_EMBEDDER") == "1"


def get_service_embedder():
    """Per-image embedder for the API/demo: FaceNet if weights available, else Simple."""
    if not _force_offline():
        try:
            return FaceNetEmbedder(), "facenet"
        except Exception:  # noqa: BLE001
            pass
    return SimpleEmbedder(), "simple"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def get_embedder(source: str = "auto", train_images: np.ndarray | None = None):
    """Return (embedder, source_str).

    source: 'facenet' | 'eigenface' | 'auto'.
    'auto' tries FaceNet, falls back to eigenface (fitted on train_images) if the
    pretrained weights cannot be downloaded.
    """
    if source in ("facenet", "auto"):
        if source == "auto" and _force_offline():
            emb = EigenfaceEmbedder()
            if train_images is not None:
                emb.fit(train_images)
            return emb, "eigenface"
        try:
            return FaceNetEmbedder(), "facenet"
        except Exception as e:  # noqa: BLE001 — any failure -> honest fallback
            if source == "facenet":
                raise
            print(f"[embed] FaceNet weights unavailable ({type(e).__name__}); "
                  f"falling back to offline eigenface embedder.")
    emb = EigenfaceEmbedder()
    if train_images is not None:
        emb.fit(train_images)
    return emb, "eigenface"


if __name__ == "__main__":
    from ..utils import data
    X, y, _ = data.load_dataset("orl", n_subjects=4)
    emb, src = get_embedder("auto", train_images=X)
    E = emb.embed(X)
    print(f"source={src} embedding={E.shape} dtype={E.dtype}")
