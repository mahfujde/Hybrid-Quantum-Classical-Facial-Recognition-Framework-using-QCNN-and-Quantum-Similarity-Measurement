"""
Phase 1 — Classical baseline.

Pipeline:  face crop -> (FaceNet | eigenface) embedding -> {kNN|SVM} closed-set
classifier  +  cosine-similarity verification.

Produces REAL measured numbers and writes them to docs/baseline_results.md and
results/baseline_<dataset>.json. The embedding source (facenet vs offline eigenface)
is recorded in every artefact so no figure is ambiguous.
"""
from __future__ import annotations
import json
import time
import numpy as np

from .. import config
from ..utils import data
from . import embed as _embed


# ---------------------------------------------------------------------------
# shared data prep (reused by the Phase 5 benchmark for a fair comparison)
# ---------------------------------------------------------------------------
def prepare_data(dataset: str, n_subjects: int, source: str = "auto", test_size=0.3):
    """Load a dataset subset, make a stratified split, embed both halves with the
    SAME embedder. Returns a dict of arrays + metadata."""
    from sklearn.model_selection import train_test_split
    X, y, names = data.load_dataset(dataset, n_subjects=n_subjects)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=config.SEED)
    embedder, src = _embed.get_embedder(source, train_images=Xtr)
    etr, ete = embedder.embed(Xtr), embedder.embed(Xte)
    return {
        "dataset": dataset, "names": names, "embedding_source": src,
        "embedder_name": getattr(embedder, "name", src),
        "X_train": Xtr, "X_test": Xte, "y_train": ytr, "y_test": yte,
        "emb_train": etr, "emb_test": ete, "embed_dim": etr.shape[1],
    }


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def _cosine_matrix(a, b):
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return an @ bn.T


def verification_metrics(emb, y):
    """Cosine-similarity face verification on all pairs. Returns same/diff mean
    similarity and best-threshold accuracy + a simple AUC."""
    S = _cosine_matrix(emb, emb)
    n = len(y)
    iu = np.triu_indices(n, k=1)
    sims = S[iu]
    same = (y[iu[0]] == y[iu[1]])
    same_sim, diff_sim = sims[same], sims[~same]
    # best-threshold accuracy
    ths = np.linspace(sims.min(), sims.max(), 200)
    accs = [((sims[same] >= t).mean() * same.mean() +
             (sims[~same] < t).mean() * (~same).mean()) for t in ths]
    best = int(np.argmax(accs))
    # AUC (rank-based)
    from sklearn.metrics import roc_auc_score
    auc = float(roc_auc_score(same.astype(int), sims)) if same.any() and (~same).any() else float("nan")
    return {
        "verif_same_mean": float(same_sim.mean()), "verif_diff_mean": float(diff_sim.mean()),
        "verif_best_threshold": float(ths[best]), "verif_best_accuracy": float(accs[best]),
        "verif_auc": auc, "n_pairs": int(len(sims)),
    }


def classify(prep: dict, clf_kind: str = config.CLASSICAL_CLF):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    etr, ete, ytr, yte = (prep["emb_train"], prep["emb_test"],
                          prep["y_train"], prep["y_test"])
    if clf_kind == "svm":
        clf = SVC(kernel="linear", C=1.0, random_state=config.SEED)
    else:
        clf = KNeighborsClassifier(n_neighbors=min(3, len(ytr)), metric="cosine")
    t0 = time.perf_counter()
    clf.fit(etr, ytr)
    train_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = clf.predict(ete)
    infer_time = (time.perf_counter() - t0) / max(1, len(ete))
    acc = accuracy_score(yte, pred)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    cm = confusion_matrix(yte, pred).tolist()
    return {
        "classifier": clf_kind, "accuracy": float(acc), "precision": float(p),
        "recall": float(r), "f1": float(f1), "confusion_matrix": cm,
        "train_time_s": float(train_time), "infer_latency_s": float(infer_time),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


def run_dataset(dataset: str, n_subjects: int = config.N_SUBJECTS, source="auto"):
    prep = prepare_data(dataset, n_subjects, source=source)
    res = {"dataset": dataset, "n_subjects": n_subjects,
           "embedding_source": prep["embedding_source"],
           "embedder_name": prep["embedder_name"], "embed_dim": prep["embed_dim"]}
    res.update(classify(prep))
    res.update(verification_metrics(np.vstack([prep["emb_train"], prep["emb_test"]]),
                                    np.concatenate([prep["y_train"], prep["y_test"]])))
    return res


def main():
    datasets = ["orl", "yale", "lfw"]
    results = []
    for ds in datasets:
        try:
            r = run_dataset(ds)
            results.append(r)
            print(f"[{ds}] source={r['embedding_source']} acc={r['accuracy']:.3f} "
                  f"f1={r['f1']:.3f} verif_auc={r['verif_auc']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"[{ds}] SKIPPED: {type(e).__name__}: {str(e)[:120]}")
    _write_report(results)
    (config.RESULTS_DIR / "baseline_all.json").write_text(json.dumps(results, indent=2))


def _write_report(results: list[dict]):
    src = results[0]["embedding_source"] if results else "n/a"
    lines = [
        "# Phase 1 — Classical Baseline Results",
        "",
        "> RESEARCH/EDUCATION prototype. Numbers below were produced by "
        "`python -m QuantumFace.classical.baseline` on the machine that generated "
        "this file. Every value traces to that run.",
        "",
        f"**Embedding source used:** `{src}`  ",
        "- `facenet` = pretrained FaceNet InceptionResnetV1 (VGGFace2), 512-dim.",
        "- `eigenface` = offline PCA-on-pixels fallback, used when the FaceNet "
        "pretrained weights could not be downloaded (GitHub release CDN blocked). "
        "**If you see `eigenface` here, the 'Classical CNN' framing does not apply — "
        "these are classical eigenface numbers; re-run on a networked machine for FaceNet.**",
        "",
        "## Closed-set recognition + verification",
        "",
        "| Dataset | Src | Dim | Clf | Acc | Prec | Rec | F1 | Verif AUC | same/diff cos |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r['embedding_source']} | {r['embed_dim']} | "
            f"{r['classifier']} | {r['accuracy']:.3f} | {r['precision']:.3f} | "
            f"{r['recall']:.3f} | {r['f1']:.3f} | {r['verif_auc']:.3f} | "
            f"{r['verif_same_mean']:.3f}/{r['verif_diff_mean']:.3f} |")
    lines += ["",
              "Reproduce: `python -m QuantumFace.classical.baseline`",
              f"(config: N_SUBJECTS={config.N_SUBJECTS}, SEED={config.SEED})", ""]
    (config.DOCS_DIR / "baseline_results.md").write_text("\n".join(lines))
    print(f"[baseline] wrote {config.DOCS_DIR/'baseline_results.md'}")


if __name__ == "__main__":
    main()
