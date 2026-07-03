"""
Face detection + alignment (Phase 1).

For real photographs / webcam frames we use MTCNN (bundled with facenet-pytorch),
which returns an aligned FACE_SIZE x FACE_SIZE crop ready for FaceNet. An OpenCV
Haar-cascade detector is provided as a lightweight, weight-free alternative.

The benchmark datasets (ORL/Yale/LFW) are already tightly-cropped face images, so for
those we simply resize to FACE_SIZE (align_prealigned). Detection matters for the
live demo, not for the fixed evaluation subsets.
"""
from __future__ import annotations
import numpy as np

from .. import config


def align_prealigned(image: np.ndarray) -> np.ndarray:
    """Resize an already-cropped grayscale face (H,W in [0,1]) to FACE_SIZE square."""
    from PIL import Image
    pil = Image.fromarray((np.clip(image, 0, 1) * 255).astype(np.uint8))
    pil = pil.resize((config.FACE_SIZE, config.FACE_SIZE), Image.BILINEAR)
    return np.asarray(pil, dtype=np.float32) / 255.0


def detect_haar(image_bgr: np.ndarray):
    """OpenCV Haar-cascade face detection. Returns list of (x, y, w, h)."""
    import cv2
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray.astype(np.uint8), 1.1, 4)
    return [tuple(int(v) for v in f) for f in faces]


class MTCNNAligner:
    """MTCNN detection + alignment to a FaceNet-ready crop. Lazy-loaded."""

    def __init__(self):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(image_size=config.FACE_SIZE, margin=0, post_process=False)

    def align(self, image_rgb: np.ndarray):
        """image_rgb: (H,W,3) uint8. Returns aligned (FACE_SIZE,FACE_SIZE) grayscale
        float [0,1], or None if no face found."""
        from PIL import Image
        pil = Image.fromarray(image_rgb.astype(np.uint8))
        face = self.mtcnn(pil)  # torch tensor (3,S,S) or None
        if face is None:
            return None
        arr = face.permute(1, 2, 0).cpu().numpy()
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        gray = arr.mean(axis=2)
        return gray.astype(np.float32)
