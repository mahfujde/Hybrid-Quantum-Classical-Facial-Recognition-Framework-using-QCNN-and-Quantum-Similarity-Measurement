"""
Optional AI feature — Explainable AI (XAI) for the recognition pipeline.

Two model-agnostic, offline explanations:

  1. occlusion_saliency(): slide a grey patch over the face; wherever hiding pixels most
     drops the match similarity, those pixels mattered most. Produces a saliency heatmap
     over the image — the classic occlusion-sensitivity method (Zeiler & Fergus 2014).

  2. quantum_feature_attribution(): which PCA components (i.e. which angle-encoded qubits)
     drive a QCNN prediction? We use permutation importance — shuffle one input feature
     across the test set and measure the accuracy drop. Larger drop = more important
     qubit/feature. This attributes the quantum model's behaviour to specific inputs
     without needing gradients through the circuit.

Both are honest: they explain THIS model on THIS data, with measured numbers.
"""
from __future__ import annotations
import numpy as np

from .. import config


# ---------------------------------------------------------------------------
# 1. occlusion saliency for embedding-based recognition
# ---------------------------------------------------------------------------
def occlusion_saliency(image: np.ndarray, reference: np.ndarray, embedder,
                       patch=12, stride=6) -> np.ndarray:
    """Heatmap: how much occluding each region reduces cosine similarity to `reference`.

    image, reference : grayscale [0,1] arrays. embedder : object with .embed(NxHxW).
    Returns a heatmap the size of `image` (higher = more important to the match)."""
    base = embedder.embed(image[None])[0]
    ref = embedder.embed(reference[None])[0]

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    base_sim = cos(base, ref)
    h, w = image.shape
    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)
    fill = float(image.mean())
    for y in range(0, h - patch + 1, stride):
        for x in range(0, w - patch + 1, stride):
            occ = image.copy()
            occ[y:y + patch, x:x + patch] = fill
            sim = cos(embedder.embed(occ[None])[0], ref)
            heat[y:y + patch, x:x + patch] += (base_sim - sim)
            counts[y:y + patch, x:x + patch] += 1.0
    heat /= np.maximum(counts, 1.0)
    if heat.max() > heat.min():
        heat = (heat - heat.min()) / (heat.max() - heat.min())
    return heat


# ---------------------------------------------------------------------------
# 2. permutation feature attribution for the QCNN
# ---------------------------------------------------------------------------
def quantum_feature_attribution(model, X, y, n_classes, n_repeats=3, seed=config.SEED):
    """Permutation importance of each angle-encoded feature (qubit) for a trained QCNN.

    model : dict from quantum.train (has circuit + params + mode).
    Returns per-feature mean accuracy drop when that feature is shuffled."""
    from ..quantum.train import predict
    rng = np.random.default_rng(seed)
    base_pred, _ = predict(model, X, n_classes)
    base_acc = float((base_pred == y).mean())
    n_feat = X.shape[1]
    importances = np.zeros(n_feat, dtype=float)
    for j in range(n_feat):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred, _ = predict(model, Xp, n_classes)
            drops.append(base_acc - float((pred == y).mean()))
        importances[j] = float(np.mean(drops))
    return {"base_accuracy": base_acc,
            "feature_importance": importances.tolist(),
            "most_important_qubit": int(np.argmax(importances)),
            "n_features": int(n_feat), "n_repeats": n_repeats}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    from ..utils import data
    from ..classical.embed import EigenfaceEmbedder
    X, y, _ = data.load_dataset("orl", n_subjects=4)
    emb = EigenfaceEmbedder(dim=32).fit(X)
    heat = occlusion_saliency(X[1], X[0], emb, patch=16, stride=12)
    print(f"occlusion saliency: heatmap {heat.shape} peak at "
          f"{np.unravel_index(int(heat.argmax()), heat.shape)} (row,col)")
