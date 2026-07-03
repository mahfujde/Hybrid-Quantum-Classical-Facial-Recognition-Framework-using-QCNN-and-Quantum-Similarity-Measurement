"""
Optional AI feature — presentation-attack detection (anti-spoofing) + liveness.

OFFLINE & CLASSICAL: a genuine (live-camera) face and a spoof (printed photo or replayed
screen) differ in texture and frequency content — prints lose high-frequency detail and
add paper/screen texture; replays add moire and recompression artefacts. We measure two
weight-free cues:

  * high-frequency energy ratio (FFT)  — spoofs typically lose fine detail;
  * local-texture sharpness (variance of Laplacian) — spoofs are often smoother/blurrier.

This is a lightweight research baseline, NOT a production PAD system (which would use a
trained CNN + depth/IR). Because this repo has no real spoof captures, we validate the
detector honestly by simulating prints/replays from genuine images (blur + JPEG
recompression + mild downscale) and reporting the measured genuine-vs-spoof separation.

Liveness (`blink_liveness`) needs a short frame SEQUENCE: it tracks eye-region intensity
variance over time; a live subject blinks/moves, a static photo does not. A single image
cannot prove liveness — the API says so rather than faking it.
"""
from __future__ import annotations
import io
import numpy as np


# ---------------------------------------------------------------------------
# single-image spoof cues (weight-free)
# ---------------------------------------------------------------------------
def _laplacian_var(gray: np.ndarray) -> float:
    """Variance of the Laplacian — a classic focus/sharpness measure."""
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    g = gray.astype(np.float32)
    from numpy.lib.stride_tricks import sliding_window_view
    if min(g.shape) < 3:
        return 0.0
    w = sliding_window_view(g, (3, 3))
    lap = (w * k).sum(axis=(-1, -2))
    return float(lap.var())


def _high_freq_ratio(gray: np.ndarray) -> float:
    """Fraction of spectral energy above a mid-frequency radius (FFT)."""
    g = gray.astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g))
    mag = np.abs(F) ** 2
    h, w = g.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r0 = 0.25 * min(h, w)
    total = mag.sum() + 1e-9
    return float(mag[r > r0].sum() / total)


def liveness_score(gray: np.ndarray) -> dict:
    """Return spoof cues + a combined 'genuineness' score in [0,1] for one image.

    Higher = more likely a genuine live capture. The threshold is dataset-dependent;
    calibrate with `evaluate_detector`."""
    lap = _laplacian_var(gray)
    hf = _high_freq_ratio(gray)
    # squashing constants chosen so typical genuine grayscale faces land near ~0.5-0.9
    sharp = 1.0 - np.exp(-lap / 5e-3)
    score = float(0.5 * sharp + 0.5 * min(1.0, hf / 0.15))
    return {"laplacian_var": lap, "high_freq_ratio": hf, "genuineness": score}


# ---------------------------------------------------------------------------
# spoof simulation (so we can measure the detector without real attack captures)
# ---------------------------------------------------------------------------
def simulate_spoof(gray: np.ndarray, jpeg_quality=25, blur=1.2) -> np.ndarray:
    """Emulate a print/replay attack: blur + JPEG recompression + slight downscale."""
    from PIL import Image, ImageFilter
    pil = Image.fromarray((np.clip(gray, 0, 1) * 255).astype(np.uint8))
    pil = pil.filter(ImageFilter.GaussianBlur(blur))
    small = pil.resize((max(8, pil.width // 2), max(8, pil.height // 2)), Image.BILINEAR)
    pil = small.resize(pil.size, Image.BILINEAR)
    buf = io.BytesIO(); pil.save(buf, format="JPEG", quality=jpeg_quality)
    out = Image.open(io.BytesIO(buf.getvalue())).convert("L")
    return np.asarray(out, dtype=np.float32) / 255.0


def evaluate_detector(images: np.ndarray) -> dict:
    """Measure genuine-vs-simulated-spoof separation and best-threshold accuracy."""
    genuine = [liveness_score(im)["genuineness"] for im in images]
    spoof = [liveness_score(simulate_spoof(im))["genuineness"] for im in images]
    g, s = np.array(genuine), np.array(spoof)
    scores = np.concatenate([g, s])
    labels = np.concatenate([np.ones_like(g), np.zeros_like(s)])
    ths = np.linspace(scores.min(), scores.max(), 200)
    accs = [((g >= t).mean() * 0.5 + (s < t).mean() * 0.5) for t in ths]
    best = int(np.argmax(accs))
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, scores))
    except Exception:
        auc = float("nan")
    return {"genuine_mean": float(g.mean()), "spoof_mean": float(s.mean()),
            "separation": float(g.mean() - s.mean()),
            "best_threshold": float(ths[best]), "best_accuracy": float(accs[best]),
            "auc": auc, "n": int(len(images))}


# ---------------------------------------------------------------------------
# liveness over a frame sequence (needs motion/blink)
# ---------------------------------------------------------------------------
def blink_liveness(frames: np.ndarray, eye_band=(0.2, 0.45)) -> dict:
    """Temporal liveness cue over a sequence of aligned grayscale frames (T,H,W).

    Tracks intensity variance in the eye band across time; a blink/motion produces a
    temporal change a static photo lacks. Returns a 'live' flag + the motion energy."""
    frames = np.asarray(frames, dtype=np.float32)
    if frames.ndim != 3 or len(frames) < 2:
        return {"live": None, "reason": "need >= 2 aligned frames; single image cannot "
                "prove liveness", "temporal_motion": None}
    h = frames.shape[1]
    band = frames[:, int(eye_band[0] * h):int(eye_band[1] * h), :]
    per_frame = band.reshape(len(band), -1).mean(axis=1)
    motion = float(np.var(per_frame)) + float(np.mean(np.var(np.diff(band, axis=0), axis=(1, 2))))
    return {"live": bool(motion > 1e-4), "temporal_motion": motion,
            "n_frames": int(len(frames))}


if __name__ == "__main__":
    from ..utils import data
    X, _, _ = data.load_dataset("orl", n_subjects=6)
    r = evaluate_detector(X)
    print(f"anti-spoof (genuine vs simulated print/replay): "
          f"genuine={r['genuine_mean']:.3f} spoof={r['spoof_mean']:.3f} "
          f"sep={r['separation']:.3f} acc={r['best_accuracy']:.3f} auc={r['auc']:.3f}")
    live = blink_liveness(np.stack([X[0], X[0]]))  # identical frames -> no motion
    print("liveness (2 identical frames):", live)
