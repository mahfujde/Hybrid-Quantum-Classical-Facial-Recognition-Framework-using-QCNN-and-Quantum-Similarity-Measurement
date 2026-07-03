"""
Dataset loading for QuantumFace.

Three public research datasets, all under their stated research-use terms:
  * ORL / AT&T  (local PGM in QuantumFace/dataset/att_faces)
  * Yale Faces  (local, QuantumFace/dataset/yalefaces)
  * LFW subset  (fetched via scikit-learn's fetch_lfw_people on first use)

Every loader returns a uniform (images, labels, names) tuple where
  images : float32 array, shape (N, H, W), pixel range [0, 1], grayscale
  labels : int array, shape (N,)
  names  : list[str] mapping label id -> human-readable identity

No face images of real people are used outside these licensed research sets.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config


def _read_gray(path: str, size: tuple[int, int] | None = None) -> np.ndarray:
    img = Image.open(path).convert("L")
    if size is not None:
        img = img.resize(size, Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_orl(n_subjects: int | None = None, size=(112, 92)):
    """ORL/AT&T: 40 subjects x 10 images, 112x92 PGM."""
    root = Path(config.ORL_DIR)
    subjects = sorted([d for d in root.glob("s*") if d.is_dir()],
                      key=lambda p: int(p.name[1:]))
    if n_subjects:
        subjects = subjects[:n_subjects]
    imgs, labels, names = [], [], []
    for lbl, sdir in enumerate(subjects):
        names.append(sdir.name)
        for f in sorted(sdir.glob("*.pgm")):
            imgs.append(_read_gray(str(f), size))
            labels.append(lbl)
    return np.stack(imgs), np.asarray(labels), names


def load_yale(n_subjects: int | None = None, size=(112, 92)):
    """Yale Faces: files named subject01.<expr>."""
    root = Path(config.YALE_DIR)
    files = [f for f in root.iterdir() if f.is_file() and f.name.startswith("subject")]
    by_subj: dict[str, list[str]] = {}
    for f in files:
        subj = f.name.split(".")[0]
        by_subj.setdefault(subj, []).append(str(f))
    subj_ids = sorted(by_subj)
    if n_subjects:
        subj_ids = subj_ids[:n_subjects]
    imgs, labels, names = [], [], []
    for lbl, subj in enumerate(subj_ids):
        names.append(subj)
        for f in sorted(by_subj[subj]):
            imgs.append(_read_gray(f, size))
            labels.append(lbl)
    return np.stack(imgs), np.asarray(labels), names


def load_lfw(n_subjects: int | None = None, min_faces_per_person=20, size=(112, 92)):
    """Small LFW subset via scikit-learn (downloads to ~/scikit_learn_data on first call)."""
    from sklearn.datasets import fetch_lfw_people
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                           resize=0.5, color=False)
    X = lfw.images.astype(np.float32)
    X = (X - X.min()) / (X.max() - X.min() + 1e-8)
    y = lfw.target.astype(int)
    names = list(lfw.target_names)
    if n_subjects:
        keep = np.isin(y, np.arange(n_subjects))
        X, y = X[keep], y[keep]
        names = names[:n_subjects]
    # resize to common size
    out = np.stack([_resize_arr(im, size) for im in X])
    return out, y, names


def _resize_arr(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    im = Image.fromarray((arr * 255).astype(np.uint8))
    im = im.resize(size, Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


LOADERS = {"orl": load_orl, "yale": load_yale, "lfw": load_lfw}


def load_dataset(name: str, n_subjects: int | None = None, size=(112, 92)):
    name = name.lower()
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; choose from {list(LOADERS)}")
    return LOADERS[name](n_subjects=n_subjects, size=size)


if __name__ == "__main__":
    for ds in ("orl", "yale"):
        X, y, names = load_dataset(ds, n_subjects=4)
        print(f"{ds}: X={X.shape} labels={np.bincount(y)} names={names}")
