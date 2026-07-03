"""
Longer / higher-qubit training experiments, checkpointed so they can run in short
chunks (this environment can't keep a long job alive, so we resume across calls).

    python -m QuantumFace.experiments.long_run step <ds> <mode> <nq> <total_steps>
        run ONE chunk of training (repeat until it reports done)
    python -m QuantumFace.experiments.long_run finalize <ds> <mode> <nq> <total_steps>
        evaluate the finished checkpoint and append to results/long_run.json

Every number written here is measured on the local simulator; nothing is projected.
"""
from __future__ import annotations
import json
import os
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from .. import config
from ..classical.baseline import prepare_data
from ..quantum.reduce import PCAReducer
from ..quantum import train as T

CACHE = config.RESULTS_DIR / "_long_cache"
CACHE.mkdir(exist_ok=True)
OUT = config.RESULTS_DIR / "long_run.json"


def _feats(ds: str, nq: int):
    fp = CACHE / f"{ds}_{nq}_feats.npz"
    if fp.exists():
        d = np.load(fp)
        return d["Xtr"], d["Xte"], d["ytr"], d["yte"], str(d["src"])
    prep = prepare_data(ds, config.N_SUBJECTS, source="auto")
    red = PCAReducer(dim=nq).fit(prep["emb_train"])
    Xtr, Xte = red.transform(prep["emb_train"]), red.transform(prep["emb_test"])
    np.savez(fp, Xtr=Xtr, Xte=Xte, ytr=prep["y_train"], yte=prep["y_test"],
             src=prep["embedding_source"])
    return Xtr, Xte, prep["y_train"], prep["y_test"], prep["embedding_source"]


def _ckpt(ds, mode, nq, steps):
    return CACHE / f"{ds}_{mode}_{nq}q_{steps}s.pkl"


def step(ds, mode, nq, total):
    Xtr, _, ytr, _, src = _feats(ds, nq)
    r = T.train_resumable(Xtr, ytr, nq, config.N_SUBJECTS, mode, total,
                          _ckpt(ds, mode, nq, total),
                          chunk=int(os.environ.get("QF_CHUNK", 50)))
    print(f"[{ds} {mode} {nq}q] step {r['step']}/{total} "
          f"loss={r['loss']:.4f} chunk={r['chunk_time_s']:.1f}s done={r['done']}")
    return r["done"]


def finalize(ds, mode, nq, total):
    Xtr, Xte, ytr, yte, src = _feats(ds, nq)
    model = T.model_from_ckpt(_ckpt(ds, mode, nq, total))
    ev = T.evaluate(model, Xte, yte, config.N_SUBJECTS)
    rec = {"dataset": ds, "mode": mode, "n_qubits": nq, "total_steps": total,
           "embedding_source": src, "accuracy": ev["accuracy"], "f1": ev["f1"],
           "n_params": model["n_params"], "loss_end": model["history"][-1],
           "loss_start": model["history"][0], "n_steps_recorded": len(model["history"])}
    data = json.loads(OUT.read_text()) if OUT.exists() else []
    data = [d for d in data if not (d["dataset"] == ds and d["mode"] == mode
            and d["n_qubits"] == nq and d["total_steps"] == total)]
    data.append(rec)
    OUT.write_text(json.dumps(data, indent=2))
    print(f"[finalize {ds} {mode} {nq}q {total}s] acc={ev['accuracy']:.3f} "
          f"f1={ev['f1']:.3f} loss {rec['loss_start']:.3f}->{rec['loss_end']:.3f} "
          f"params={model['n_params']} src={src}")


def main(argv):
    cmd = argv[1]
    ds, mode, nq, total = argv[2], argv[3], int(argv[4]), int(argv[5])
    if cmd == "step":
        done = step(ds, mode, nq, total)
        sys.exit(0 if done else 3)   # exit 3 = more chunks needed
    elif cmd == "finalize":
        finalize(ds, mode, nq, total)


if __name__ == "__main__":
    main(sys.argv)
