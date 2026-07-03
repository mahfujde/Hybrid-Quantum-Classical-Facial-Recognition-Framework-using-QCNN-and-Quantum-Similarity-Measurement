"""
Phase 4 — training the quantum models.

Two variants trained by classical optimisation of the variational parameters:
  * pure QCNN  : QCNN -> pure_readout (minimal classical head)
  * hybrid QCNN: QCNN -> trainable classical linear head (hybrid_readout)

Both consume the SAME PCA-reduced features and the SAME train/test split as the
classical baseline, so the Phase 5 comparison is fair. Training time, parameter count
and the loss curve are logged for every run.
"""
from __future__ import annotations
import time
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .. import config
from . import qcnn as Q


def _cross_entropy(prob, y):
    return -pnp.log(prob[int(y)] + 1e-9)


def train_pure(Xtr, ytr, n_qubits, n_classes, steps=config.TRAIN_STEPS, lr=0.15, tag="pure"):
    circ = Q.build_qcnn(n_qubits=n_qubits, n_classes=n_classes)
    conv, pool = Q.init_conv_pool(n_qubits)
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(conv, pool):
        loss = 0.0
        for x, y in zip(Xtr, ytr):
            loss = loss + _cross_entropy(Q.pure_readout(circ(x, conv, pool), n_classes), y)
        return loss / len(Xtr)

    hist, t0 = [], time.perf_counter()
    for s in range(steps):
        (conv, pool), c = opt.step_and_cost(cost, conv, pool)
        hist.append(float(c))
    train_time = time.perf_counter() - t0
    params = {"conv": conv, "pool": pool}
    return {"circuit": circ, "params": params, "history": hist,
            "train_time_s": train_time,
            "n_params": int(conv.size + pool.size), "mode": "pure"}


def train_hybrid(Xtr, ytr, n_qubits, n_classes, steps=config.TRAIN_STEPS, lr=0.1, tag="hybrid"):
    circ = Q.build_qcnn(n_qubits=n_qubits, n_classes=n_classes)
    conv, pool = Q.init_conv_pool(n_qubits)
    n_measure = Q.n_survivors(n_qubits)
    w, b = Q.init_hybrid_head(n_measure, n_classes)
    opt = qml.AdamOptimizer(stepsize=lr)

    def cost(conv, pool, w, b):
        loss = 0.0
        for x, y in zip(Xtr, ytr):
            loss = loss + _cross_entropy(Q.hybrid_readout(circ(x, conv, pool), w, b), y)
        return loss / len(Xtr)

    hist, t0 = [], time.perf_counter()
    for s in range(steps):
        (conv, pool, w, b), c = opt.step_and_cost(cost, conv, pool, w, b)
        hist.append(float(c))
    train_time = time.perf_counter() - t0
    params = {"conv": conv, "pool": pool, "w": w, "b": b}
    return {"circuit": circ, "params": params, "history": hist,
            "train_time_s": train_time,
            "n_params": int(conv.size + pool.size + w.size + b.size), "mode": "hybrid"}


def train_resumable(Xtr, ytr, n_qubits, n_classes, mode, total_steps, ckpt_path,
                    chunk=50, lr=None):
    """Checkpointed training so long runs can span multiple short sessions.

    Loads state from ckpt_path if present, runs up to `chunk` more steps (until
    total_steps reached), saves state, and returns progress. The optimiser state is
    pickled so Adam momentum survives across chunks.
    """
    import pickle
    from pathlib import Path
    Xtr = np.asarray(Xtr, dtype=np.float32)
    ckpt = Path(ckpt_path)
    circ = Q.build_qcnn(n_qubits=n_qubits, n_classes=n_classes)

    if ckpt.exists():
        st = pickle.loads(ckpt.read_bytes())
    else:
        conv, pool = Q.init_conv_pool(n_qubits)
        st = {"conv": conv, "pool": pool, "step": 0, "history": [],
              "mode": mode, "n_qubits": n_qubits, "n_classes": n_classes,
              "opt": qml.AdamOptimizer(stepsize=lr or (0.15 if mode == "pure" else 0.1))}
        if mode == "hybrid":
            w, b = Q.init_hybrid_head(Q.n_survivors(n_qubits), n_classes)
            st["w"], st["b"] = w, b

    opt = st["opt"]
    t0 = time.perf_counter()
    target = min(st["step"] + chunk, total_steps)

    if mode == "pure":
        def cost(conv, pool):
            loss = 0.0
            for x, y in zip(Xtr, ytr):
                loss = loss + _cross_entropy(Q.pure_readout(circ(x, conv, pool), n_classes), y)
            return loss / len(Xtr)
        while st["step"] < target:
            (st["conv"], st["pool"]), c = opt.step_and_cost(cost, st["conv"], st["pool"])
            st["history"].append(float(c)); st["step"] += 1
    else:
        def cost(conv, pool, w, b):
            loss = 0.0
            for x, y in zip(Xtr, ytr):
                loss = loss + _cross_entropy(Q.hybrid_readout(circ(x, conv, pool), w, b), y)
            return loss / len(Xtr)
        while st["step"] < target:
            (st["conv"], st["pool"], st["w"], st["b"]), c = opt.step_and_cost(
                cost, st["conv"], st["pool"], st["w"], st["b"])
            st["history"].append(float(c)); st["step"] += 1

    st["opt"] = opt
    ckpt.write_bytes(pickle.dumps(st))
    return {"done": st["step"] >= total_steps, "step": st["step"],
            "total_steps": total_steps, "chunk_time_s": time.perf_counter() - t0,
            "loss": st["history"][-1] if st["history"] else None}


def model_from_ckpt(ckpt_path):
    """Reconstruct a usable model dict (with rebuilt circuit) from a checkpoint."""
    import pickle
    from pathlib import Path
    st = pickle.loads(Path(ckpt_path).read_bytes())
    circ = Q.build_qcnn(n_qubits=st["n_qubits"], n_classes=st["n_classes"])
    params = {"conv": st["conv"], "pool": st["pool"]}
    if st["mode"] == "hybrid":
        params["w"], params["b"] = st["w"], st["b"]
    n_params = int(st["conv"].size + st["pool"].size +
                   (st["w"].size + st["b"].size if st["mode"] == "hybrid" else 0))
    return {"circuit": circ, "params": params, "mode": st["mode"],
            "history": st["history"], "n_params": n_params,
            "train_time_s": None}


def predict(model, X, n_classes):
    circ, p = model["circuit"], model["params"]
    preds, t0 = [], time.perf_counter()
    for x in X:
        m = circ(x, p["conv"], p["pool"])
        if model["mode"] == "pure":
            prob = Q.pure_readout(m, n_classes)
        else:
            prob = Q.hybrid_readout(m, p["w"], p["b"])
        preds.append(int(pnp.argmax(prob)))
    latency = (time.perf_counter() - t0) / max(1, len(X))
    return np.array(preds), latency


def evaluate(model, Xte, yte, n_classes):
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix)
    pred, latency = predict(model, Xte, n_classes)
    p, r, f1, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(yte, pred)), "precision": float(p),
            "recall": float(r), "f1": float(f1),
            "confusion_matrix": confusion_matrix(yte, pred).tolist(),
            "infer_latency_s": float(latency)}
