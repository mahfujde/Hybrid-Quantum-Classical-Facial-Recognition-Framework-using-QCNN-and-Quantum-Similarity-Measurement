"""
Optional AI feature — continual (incremental) learning of new identities.

Enrolling a new person should NOT require retraining the whole model. A prototype /
nearest-class-mean (NCM) classifier stores one mean embedding ("prototype") per identity
and classifies by cosine distance to prototypes. Adding a new identity is O(1): compute
its prototype and append it — no gradient steps, no touching existing classes. This is a
standard, well-understood continual-learning baseline (Mensink et al. 2013; Rebuffi et
al. iCaRL 2017 use NCM at inference).

`measure_forgetting()` quantifies catastrophic forgetting: it enrols identities in
batches and, after each batch, measures accuracy on the ORIGINAL identities. Because NCM
prototypes are independent per class, forgetting is ~0 by construction — we report the
measured number to prove it rather than assert it.
"""
from __future__ import annotations
import numpy as np

from .. import config


class PrototypeClassifier:
    """Nearest-class-mean classifier with O(1) incremental enrolment."""

    def __init__(self):
        self.prototypes: dict[int, np.ndarray] = {}
        self._names: dict[int, str] = {}

    @staticmethod
    def _norm(v):
        return v / (np.linalg.norm(v) + 1e-9)

    def enroll(self, label: int, embeddings: np.ndarray, name: str | None = None):
        """Add/replace one identity from one or more example embeddings."""
        proto = np.asarray(embeddings, dtype=np.float32).reshape(-1, embeddings.shape[-1]).mean(0)
        self.prototypes[label] = self._norm(proto)
        if name:
            self._names[label] = name
        return self

    def enroll_batch(self, X: np.ndarray, y: np.ndarray):
        for lbl in np.unique(y):
            self.enroll(int(lbl), X[y == lbl])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        labels = list(self.prototypes)
        P = np.stack([self.prototypes[l] for l in labels])
        Xn = np.stack([self._norm(x) for x in np.asarray(X, dtype=np.float32)])
        sims = Xn @ P.T
        return np.array([labels[i] for i in sims.argmax(1)])

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())

    @property
    def n_classes(self):
        return len(self.prototypes)


def measure_forgetting(dataset="orl", n_subjects=8, batch=2, test_size=0.3):
    """Enrol identities in batches; after each batch measure accuracy on the FIRST batch's
    identities (the 'old' task). Returns the accuracy trajectory + a forgetting metric."""
    from sklearn.model_selection import train_test_split
    from ..utils import data
    from ..classical.embed import EigenfaceEmbedder

    X, y, _ = data.load_dataset(dataset, n_subjects=n_subjects)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.SEED)
    emb = EigenfaceEmbedder(dim=64).fit(Xtr)
    etr, ete = emb.embed(Xtr), emb.embed(Xte)

    all_labels = sorted(np.unique(y).tolist())
    first_task = all_labels[:batch]
    clf = PrototypeClassifier()
    trajectory = []
    for i in range(0, len(all_labels), batch):
        chunk = all_labels[i:i + batch]
        mask = np.isin(ytr, chunk)
        clf.enroll_batch(etr[mask], ytr[mask])
        # accuracy on the original identities only
        old_mask = np.isin(yte, first_task)
        acc_old = clf.score(ete[old_mask], yte[old_mask])
        acc_all = clf.score(ete, yte)
        trajectory.append({"classes_seen": clf.n_classes,
                           "acc_first_task": round(acc_old, 3),
                           "acc_all_seen": round(acc_all, 3)})
    first_acc = trajectory[0]["acc_first_task"]
    last_acc = trajectory[-1]["acc_first_task"]
    return {"dataset": dataset, "n_subjects": n_subjects, "batch_size": batch,
            "trajectory": trajectory,
            "forgetting": round(first_acc - last_acc, 3),
            "note": "NCM prototypes are per-class independent -> forgetting ~ 0 by design"}


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    r = measure_forgetting()
    for t in r["trajectory"]:
        print(t)
    print("forgetting (acc drop on first identities):", r["forgetting"])
