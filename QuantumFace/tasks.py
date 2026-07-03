"""
QuantumFace task runner (CI-friendly, no Make required).

Usage:
    python -m QuantumFace.tasks <command>

Commands:
    baseline    Phase 1: classical FaceNet baseline -> docs/baseline_results.md
    reduce      Phase 2: PCA reduction + encoding sanity report
    benchmark   Phase 5: three-way benchmark -> docs/benchmark_report.md
    test        Run the pytest suite
    all         baseline -> benchmark -> test
"""
from __future__ import annotations
import subprocess
import sys


def _run(mod: str):
    return subprocess.call([sys.executable, "-m", mod])


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "all"
    if cmd == "baseline":
        return _run("QuantumFace.classical.baseline")
    if cmd == "reduce":
        return _run("QuantumFace.quantum.encoding")
    if cmd == "benchmark":
        return _run("QuantumFace.benchmark")
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "QuantumFace/tests"])
    if cmd == "all":
        for m in ("QuantumFace.classical.baseline", "QuantumFace.benchmark"):
            if _run(m):
                return 1
        return subprocess.call([sys.executable, "-m", "pytest", "-q", "QuantumFace/tests"])
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
