"""
Phase 5 — the research contribution: a fair three-way benchmark.

Classical baseline  vs  Hybrid QCNN  vs  Pure QCNN, on the SAME split and features:

    face -> embedding -> [classical: classifier]      (Classical)
                      -> PCA -> QCNN -> classical head (Hybrid)
                      -> PCA -> QCNN -> softmax        (Pure)

Reports accuracy / precision / recall / F1 / confusion, training time, inference
latency (measured), parameter count, qubit count, circuit depth, and a SWAP-test
noise-robustness sweep. Every number is produced here; nothing is projected.

Because quantum simulation is slow, work is CACHED per (dataset, stage) so it can run
in small chunks and resume. Stages:

    python -m QuantumFace.benchmark base   <ds>   # classical + info-loss + SWAP + features
    python -m QuantumFace.benchmark pure   <ds>   # train+eval pure QCNN
    python -m QuantumFace.benchmark hybrid <ds>   # train+eval hybrid QCNN
    python -m QuantumFace.benchmark report        # merge caches -> report + benchmark.json
    python -m QuantumFace.benchmark all    <ds>   # base+pure+hybrid for one dataset

Env:  QF_SUBJECTS (default 4)  QF_STEPS (default 30)  QF_N_QUBITS (default 8)
      QF_DATASETS (default "orl,yale")
"""
from __future__ import annotations
import json
import os
import platform
import sys
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from . import config
from .classical.baseline import prepare_data, classify, verification_metrics
from .quantum.reduce import PCAReducer, information_loss_report
from .quantum import qcnn as Q
from .quantum import train as T
from .quantum.swap import build_swap_test

STEPS = int(os.environ.get("QF_STEPS", 30))
DATASETS = os.environ.get("QF_DATASETS", "orl,yale").split(",")
CACHE = config.RESULTS_DIR / "_bench_cache"
CACHE.mkdir(exist_ok=True)


def _swap_sweep_subsampled(feats, labels, n_feat=4, max_pairs=30):
    feats, labels = np.asarray(feats), np.asarray(labels)
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same = labels[iu[0]] == labels[iu[1]]
    idx = np.arange(len(same))
    rng = np.random.default_rng(config.SEED)
    same_i = rng.permutation(idx[same])[:max_pairs]
    diff_i = rng.permutation(idx[~same])[:max_pairs]
    out = {"n_feat": n_feat, "noise_levels": list(config.NOISE_LEVELS),
           "swap_same": [], "swap_diff": [], "swap_margin": []}
    for p in config.NOISE_LEVELS:
        fid = build_swap_test(n_feat=n_feat, noise=(p > 0), p=p)
        sm = float(np.mean([fid(feats[iu[0][k]], feats[iu[1][k]]) for k in same_i]))
        df = float(np.mean([fid(feats[iu[0][k]], feats[iu[1][k]]) for k in diff_i]))
        out["swap_same"].append(sm); out["swap_diff"].append(df)
        out["swap_margin"].append(sm - df)
    return out


def stage_base(ds: str):
    nq = config.N_QUBITS
    prep = prepare_data(ds, config.N_SUBJECTS, source="auto")
    ytr, yte = prep["y_train"], prep["y_test"]
    classical = classify(prep)
    verif = verification_metrics(np.vstack([prep["emb_train"], prep["emb_test"]]),
                                 np.concatenate([ytr, yte]))
    info = information_loss_report(prep["emb_train"], ytr, prep["emb_test"], yte, dim=nq)
    red = PCAReducer(dim=nq).fit(prep["emb_train"])
    Xtr_q, Xte_q = red.transform(prep["emb_train"]), red.transform(prep["emb_test"])
    np.savez(CACHE / f"{ds}_feats.npz", Xtr=Xtr_q, Xte=Xte_q, ytr=ytr, yte=yte)
    swap = _swap_sweep_subsampled(Xte_q, yte, n_feat=min(4, nq))
    base = {"dataset": ds, "n_subjects": config.N_SUBJECTS, "n_qubits": nq,
            "embedding_source": prep["embedding_source"], "embed_dim": prep["embed_dim"],
            "train_steps": STEPS, "info_loss": info, "circuit": Q.describe_circuit(nq),
            "classical": {**{k: classical[k] for k in ("classifier", "accuracy",
                          "precision", "recall", "f1", "train_time_s", "infer_latency_s")},
                          "n_params": None, "n_qubits": 0, "depth": 0,
                          "confusion_matrix": classical["confusion_matrix"]},
            "verification": verif, "swap_noise_sweep": swap}
    (CACHE / f"{ds}_base.json").write_text(json.dumps(base, indent=2))
    print(f"[base:{ds}] classical acc={classical['accuracy']:.3f} src={base['embedding_source']} "
          f"info_drop={info['accuracy_drop_from_reduction']:.3f}")


def _load_feats(ds):
    d = np.load(CACHE / f"{ds}_feats.npz")
    return d["Xtr"], d["Xte"], d["ytr"], d["yte"]


def stage_train(ds: str, mode: str):
    nq, nsub = config.N_QUBITS, config.N_SUBJECTS
    Xtr, Xte, ytr, yte = _load_feats(ds)
    trainer = T.train_pure if mode == "pure" else T.train_hybrid
    model = trainer(Xtr, ytr, nq, nsub, steps=STEPS)
    ev = T.evaluate(model, Xte, yte, nsub)
    depth = Q.describe_circuit(nq)["depth"]
    rec = {**ev, "train_time_s": model["train_time_s"], "n_params": model["n_params"],
           "n_qubits": nq, "depth": depth, "history": model["history"], "mode": mode}
    (CACHE / f"{ds}_{mode}.json").write_text(json.dumps(rec, indent=2))
    print(f"[{mode}:{ds}] acc={ev['accuracy']:.3f} f1={ev['f1']:.3f} "
          f"train={model['train_time_s']:.1f}s params={model['n_params']}")


def report():
    results = []
    for ds in DATASETS:
        ds = ds.strip()
        bf = CACHE / f"{ds}_base.json"
        if not bf.exists():
            continue
        base = json.loads(bf.read_text())
        models = {"classical": base.pop("classical")}
        for mode in ("hybrid", "pure"):
            mf = CACHE / f"{ds}_{mode}.json"
            if mf.exists():
                models[f"{mode}_qcnn"] = json.loads(mf.read_text())
        base["models"] = models
        results.append(base)
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "simulator_only": True}
    payload = {"env": env, "results": results}
    (config.RESULTS_DIR / "benchmark.json").write_text(json.dumps(payload, indent=2))
    _write_report(payload)


def _write_report(payload: dict):
    results, env = payload["results"], payload["env"]
    L = ["# Phase 5 — Three-Way Benchmark Report",
         "",
         "> RESEARCH/EDUCATION prototype. Every number below was produced by "
         "`python -m QuantumFace.benchmark` on the machine that generated this file, "
         "on **local simulators only**. No quantum hardware, no projected numbers.",
         "",
         f"**Environment:** Python {env['python']} · {env['platform']} · "
         f"simulator-only={env['simulator_only']}", ""]
    for r in results:
        m = r["models"]; src = r["embedding_source"]
        L += [f"## {r['dataset'].upper()}  (source=`{src}`, {r['n_subjects']} identities, "
              f"{r['n_qubits']} qubits, {r['train_steps']} train steps)", ""]
        if src == "eigenface":
            L += ["> NOTE: FaceNet weights were unavailable on this machine, so the "
                  "'Classical' row is an **eigenface** (PCA-on-pixels) baseline, not "
                  "FaceNet. Re-run on a networked machine for FaceNet numbers.", ""]
        L += ["| Model | Acc | Prec | Rec | F1 | Train(s) | Latency(s) | Params | Qubits | Depth |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for label, key in (("Classical", "classical"), ("Hybrid QCNN", "hybrid_qcnn"),
                           ("Pure QCNN", "pure_qcnn")):
            d = m.get(key)
            if not d:
                L.append(f"| {label} | TBD — pending | | | | | | | | |"); continue
            L.append(f"| {label} | {d['accuracy']:.3f} | {d['precision']:.3f} | "
                     f"{d['recall']:.3f} | {d['f1']:.3f} | {d['train_time_s']:.2f} | "
                     f"{d['infer_latency_s']:.4f} | {d['n_params']} | {d['n_qubits']} | "
                     f"{d['depth']} |")
        il = r["info_loss"]
        L += ["",
              f"**PCA reduction (isolated info loss):** {il['full_dim']}→{il['reduced_dim']} dims, "
              f"explained var {il['explained_variance_ratio']:.3f}, recon err "
              f"{il['reconstruction_error_test']:.3f}, kNN acc "
              f"{il['knn_acc_full_dim']:.3f}→{il['knn_acc_reduced_dim']:.3f} "
              f"(drop {il['accuracy_drop_from_reduction']:.3f}).", "",
              "**SWAP-test quantum similarity (noise sweep):**", "",
              "| Depol. p | same | diff | margin |", "|---|---|---|---|"]
        sw = r["swap_noise_sweep"]
        for p, s, d, mg in zip(sw["noise_levels"], sw["swap_same"], sw["swap_diff"], sw["swap_margin"]):
            L.append(f"| {p} | {s:.3f} | {d:.3f} | {mg:.3f} |")
        L += [""]
    L += ["## Reproduce", "```bash",
          "for ds in orl yale; do",
          "  python -m QuantumFace.benchmark base   $ds",
          "  python -m QuantumFace.benchmark pure   $ds",
          "  python -m QuantumFace.benchmark hybrid $ds",
          "done",
          "python -m QuantumFace.benchmark report",
          "```", f"(SEED={config.SEED}, deterministic.)", ""]
    (config.DOCS_DIR / "benchmark_report.md").write_text("\n".join(L))
    print(f"[report] wrote {config.DOCS_DIR/'benchmark_report.md'}")


def main(argv):
    cmd = argv[1] if len(argv) > 1 else "report"
    ds = argv[2] if len(argv) > 2 else "orl"
    if cmd == "base":
        stage_base(ds)
    elif cmd in ("pure", "hybrid"):
        stage_train(ds, cmd)
    elif cmd == "all":
        stage_base(ds); stage_train(ds, "pure"); stage_train(ds, "hybrid")
    elif cmd == "report":
        report()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv)
