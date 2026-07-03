"""
One-command runner to regenerate the FaceNet + LFW numbers ON A NETWORKED MACHINE.

Why this exists: the build sandbox blocks the FaceNet pretrained-weight CDN
(release-assets.githubusercontent.com), Hugging Face, and scikit-learn's LFW download,
so those numbers could not be produced there and the eigenface fallback was used. Run
this on any machine with normal internet to replace them with real FaceNet numbers.

Usage:
    pip install -r QuantumFace/requirements.txt
    python -m QuantumFace.scripts.regenerate_facenet_lfw            # ORL + Yale + LFW
    python -m QuantumFace.scripts.regenerate_facenet_lfw orl lfw    # pick datasets

What it does:
  1. Forces the FaceNet embedder (source="facenet"); errors loudly if weights can't be
     fetched, instead of silently falling back — so you know the numbers are real FaceNet.
  2. Runs the Phase 1 classical baseline and the Phase 5 three-way benchmark on each
     dataset, writing docs/baseline_results.md, docs/benchmark_report.md and
     results/*.json exactly as in-repo.
"""
from __future__ import annotations
import sys
import warnings

warnings.filterwarnings("ignore")

from .. import config
from ..classical import baseline as B
from ..classical.embed import FaceNetEmbedder


def _assert_facenet():
    try:
        FaceNetEmbedder()
        print("[ok] FaceNet pretrained weights downloaded and loaded.")
    except Exception as e:  # noqa: BLE001
        print(f"[FATAL] FaceNet weights could not be loaded: {type(e).__name__}: {e}")
        print("Run this on a machine with internet access to the GitHub release CDN.")
        sys.exit(1)


def main(argv):
    datasets = argv[1:] or ["orl", "yale", "lfw"]
    _assert_facenet()

    # Phase 1 baseline with real FaceNet
    print(f"\n== Phase 1 baseline (FaceNet) on {datasets} ==")
    results = []
    for ds in datasets:
        try:
            r = B.run_dataset(ds, source="facenet")
            results.append(r)
            print(f"  [{ds}] acc={r['accuracy']:.3f} f1={r['f1']:.3f} "
                  f"verif_auc={r['verif_auc']:.3f} (src={r['embedding_source']})")
        except Exception as e:  # noqa: BLE001
            print(f"  [{ds}] FAILED: {type(e).__name__}: {str(e)[:120]}")
    B._write_report(results)

    # Phase 5 benchmark (staged) with real FaceNet
    print("\n== Phase 5 benchmark (FaceNet) ==")
    import os
    os.environ["QF_DATASETS"] = ",".join(datasets)
    from .. import benchmark as BM
    BM.DATASETS = datasets
    for ds in datasets:
        try:
            BM.stage_base(ds)
            BM.stage_train(ds, "pure")
            BM.stage_train(ds, "hybrid")
        except Exception as e:  # noqa: BLE001
            print(f"  [{ds}] benchmark FAILED: {type(e).__name__}: {str(e)[:120]}")
    BM.report()
    print(f"\nDone. See {config.DOCS_DIR/'baseline_results.md'} and "
          f"{config.DOCS_DIR/'benchmark_report.md'}.")


if __name__ == "__main__":
    main(sys.argv)
