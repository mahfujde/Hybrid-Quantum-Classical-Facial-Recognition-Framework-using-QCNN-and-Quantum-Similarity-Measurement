"""
Phase 2 — dimensionality reduction before quantum encoding.

A 512-dim FaceNet embedding (or the offline eigenface embedding) cannot be angle-
encoded on an 8-16 qubit simulator at full fidelity, so we compress it with PCA to
N_QUBITS features. This step LOSES INFORMATION, and we measure exactly how much,
*isolated from the quantum step*:

  1. reconstruction error  ||x - PCA^{-1}(PCA(x))|| / ||x||
  2. explained-variance ratio retained
  3. the classical k-NN accuracy DROP caused by the reduction alone
     (full-dim embedding vs reduced embedding, same classifier, same split)

Reported so a reader can attribute any hybrid-model weakness to reduction vs the
quantum circuit itself.
"""
from __future__ import annotations
import numpy as np

from .. import config


class PCAReducer:
    """PCA to `dim` features, then scale to angle range [0, pi] for RY encoding."""

    def __init__(self, dim: int = config.N_QUBITS, angle_range=(0.0, np.pi)):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import MinMaxScaler
        self.dim = dim
        self._pca = PCA(n_components=dim, random_state=config.SEED)
        self._scaler = MinMaxScaler(feature_range=angle_range)
        self._fitted = False

    def fit(self, emb: np.ndarray):
        self.dim = min(self.dim, min(emb.shape) - 1) if min(emb.shape) - 1 < self.dim else self.dim
        self._pca.n_components = self.dim
        z = self._pca.fit_transform(emb)
        self._scaler.fit(z)
        self._fitted = True
        return self

    def transform(self, emb: np.ndarray) -> np.ndarray:
        z = self._pca.transform(emb)
        return self._scaler.transform(z).astype(np.float32)

    def fit_transform(self, emb):
        return self.fit(emb).transform(emb)

    @property
    def explained_variance_ratio(self) -> float:
        return float(self._pca.explained_variance_ratio_.sum())

    def reconstruction_error(self, emb: np.ndarray) -> float:
        z = self._pca.transform(emb)
        rec = self._pca.inverse_transform(z)
        num = np.linalg.norm(emb - rec, axis=1)
        den = np.linalg.norm(emb, axis=1) + 1e-9
        return float(np.mean(num / den))


def information_loss_report(emb_train, y_train, emb_test, y_test, dim=config.N_QUBITS):
    """Quantify the cost of PCA reduction, isolated from any quantum step."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score

    def _knn_acc(Xtr, Xte):
        clf = KNeighborsClassifier(n_neighbors=min(3, len(y_train)), metric="cosine")
        clf.fit(Xtr, y_train)
        return float(accuracy_score(y_test, clf.predict(Xte)))

    full_acc = _knn_acc(emb_train, emb_test)
    red = PCAReducer(dim=dim).fit(emb_train)
    red_tr, red_te = red.transform(emb_train), red.transform(emb_test)
    reduced_acc = _knn_acc(red_tr, red_te)
    return {
        "reduced_dim": red.dim,
        "explained_variance_ratio": red.explained_variance_ratio,
        "reconstruction_error_test": red.reconstruction_error(emb_test),
        "knn_acc_full_dim": full_acc,
        "knn_acc_reduced_dim": reduced_acc,
        "accuracy_drop_from_reduction": float(full_acc - reduced_acc),
        "full_dim": int(emb_train.shape[1]),
    }


if __name__ == "__main__":
    from ..classical.baseline import prepare_data
    prep = prepare_data("orl", config.N_SUBJECTS, source="eigenface")
    rep = information_loss_report(prep["emb_train"], prep["y_train"],
                                  prep["emb_test"], prep["y_test"])
    for k, v in rep.items():
        print(f"  {k}: {v}")
