"""
Optional AI feature — facial-expression ("emotion") recognition.

REAL & OFFLINE: the Yale Face Database encodes an expression in each filename
(subject01.happy, .sad, .sleepy, .surprised, .wink, .normal). We use that as ground
truth to train a genuine expression classifier on the same embeddings the rest of the
framework uses — so the accuracy reported here is measured, not assumed.

Age estimation and mask detection are provided as HONEST STUBS: doing them properly
needs pretrained models or labelled data that are not available offline in this repo.
The stubs document exactly what would be required rather than returning fake numbers.
"""
from __future__ import annotations
from pathlib import Path

import numpy as np

from .. import config
from ..utils import data as _data

# Expression suffixes in Yale that denote an emotion/expression (not lighting/glasses)
EMOTIONS = ["normal", "happy", "sad", "sleepy", "surprised", "wink"]


def load_yale_expressions(size=(112, 92)):
    """Return (images, labels, label_names) for the emotion-bearing Yale files."""
    root = Path(config.YALE_DIR)
    imgs, labels = [], []
    for f in sorted(root.iterdir()):
        if not f.is_file() or not f.name.startswith("subject"):
            continue
        parts = f.name.split(".")
        if len(parts) < 2:
            continue
        expr = parts[1]
        if expr not in EMOTIONS:
            continue
        imgs.append(_data._read_gray(str(f), size))
        labels.append(EMOTIONS.index(expr))
    return np.stack(imgs), np.asarray(labels), list(EMOTIONS)


def train_emotion_classifier(clf_kind="knn", test_size=0.3, embed_dim=64):
    """Train + evaluate an expression classifier on Yale. Returns a metrics dict."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    from ..classical.embed import EigenfaceEmbedder

    X, y, names = load_yale_expressions()
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.SEED)
    emb = EigenfaceEmbedder(dim=embed_dim).fit(Xtr)
    etr, ete = emb.embed(Xtr), emb.embed(Xte)
    clf = (SVC(kernel="linear", random_state=config.SEED) if clf_kind == "svm"
           else KNeighborsClassifier(n_neighbors=3, metric="cosine"))
    clf.fit(etr, ytr)
    pred = clf.predict(ete)
    n_classes = len(names)
    return {
        "task": "yale_expression", "classes": names, "n_classes": n_classes,
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "chance_level": round(1.0 / n_classes, 3),
        "accuracy": float(accuracy_score(yte, pred)),
        "f1_macro": float(f1_score(yte, pred, average="macro")),
        "confusion_matrix": confusion_matrix(yte, pred).tolist(),
        "classifier": clf_kind, "embedder": "eigenface", "embed_dim": embed_dim,
    }


# ---------------------------------------------------------------------------
# Honest stubs — not implemented because the offline repo lacks the required assets
# ---------------------------------------------------------------------------
class AgeEstimator:
    """Age estimation stub. Requires a pretrained age model (e.g. DEX/SSR-Net on
    IMDB-WIKI) or an age-labelled dataset — neither is available offline here."""

    available = False

    def estimate(self, image):  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "Age estimation needs a pretrained age model or age-labelled data. "
            "None ships with this offline repo. Plug in facenet-pytorch-style weights "
            "or an IMDB-WIKI-trained model to enable.")


class MaskDetector:
    """Mask detection stub. Requires a mask/no-mask labelled dataset (e.g. MaskedFace-Net)
    or a pretrained detector — not available offline here."""

    available = False

    def detect(self, image):  # pragma: no cover - intentional stub
        raise NotImplementedError(
            "Mask detection needs a mask/no-mask labelled dataset or pretrained "
            "detector. None ships with this offline repo.")


if __name__ == "__main__":
    r = train_emotion_classifier()
    print(f"Yale expression recognition: {r['n_classes']} classes {r['classes']}")
    print(f"  train={r['n_train']} test={r['n_test']} chance={r['chance_level']}")
    print(f"  accuracy={r['accuracy']:.3f} f1_macro={r['f1_macro']:.3f}")
