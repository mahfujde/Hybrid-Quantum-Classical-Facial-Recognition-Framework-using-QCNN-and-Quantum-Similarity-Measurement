# Phase 5 — Three-Way Benchmark Report

> RESEARCH/EDUCATION prototype. Every number below was produced by `python -m QuantumFace.benchmark` on the machine that generated this file, on **local simulators only**. No quantum hardware, no projected numbers.

**Environment:** Python 3.10.12 · Linux-6.8.0-124-generic-aarch64-with-glibc2.35 · simulator-only=True

## ORL  (source=`eigenface`, 4 identities, 8 qubits, 30 train steps)

> NOTE: FaceNet weights were unavailable on this machine, so the 'Classical' row is an **eigenface** (PCA-on-pixels) baseline, not FaceNet. Re-run on a networked machine for FaceNet numbers.

| Model | Acc | Prec | Rec | F1 | Train(s) | Latency(s) | Params | Qubits | Depth |
|---|---|---|---|---|---|---|---|---|---|
| Classical | 0.833 | 0.854 | 0.833 | 0.831 | 0.00 | 0.0001 | None | 0 | 0 |
| Hybrid QCNN | 0.667 | 0.812 | 0.667 | 0.631 | 19.34 | 0.0068 | 72 | 8 | 20 |
| Pure QCNN | 0.333 | 0.200 | 0.333 | 0.250 | 19.32 | 0.0099 | 52 | 8 | 20 |

**PCA reduction (isolated info loss):** 27→8 dims, explained var 0.296, recon err 0.889, kNN acc 0.833→0.333 (drop 0.500).

**SWAP-test quantum similarity (noise sweep):**

| Depol. p | same | diff | margin |
|---|---|---|---|
| 0.0 | 0.812 | 0.699 | 0.113 |
| 0.05 | 0.616 | 0.530 | 0.085 |
| 0.12 | 0.404 | 0.348 | 0.056 |
| 0.22 | 0.202 | 0.174 | 0.028 |

## YALE  (source=`eigenface`, 4 identities, 8 qubits, 30 train steps)

> NOTE: FaceNet weights were unavailable on this machine, so the 'Classical' row is an **eigenface** (PCA-on-pixels) baseline, not FaceNet. Re-run on a networked machine for FaceNet numbers.

| Model | Acc | Prec | Rec | F1 | Train(s) | Latency(s) | Params | Qubits | Depth |
|---|---|---|---|---|---|---|---|---|---|
| Classical | 0.214 | 0.300 | 0.208 | 0.196 | 0.00 | 0.0001 | None | 0 | 0 |
| Hybrid QCNN | 0.143 | 0.083 | 0.146 | 0.106 | 21.04 | 0.0064 | 72 | 8 | 20 |
| Pure QCNN | 0.357 | 0.333 | 0.354 | 0.326 | 21.21 | 0.0064 | 52 | 8 | 20 |

**PCA reduction (isolated info loss):** 30→8 dims, explained var 0.936, recon err 0.120, kNN acc 0.214→0.286 (drop -0.071).

**SWAP-test quantum similarity (noise sweep):**

| Depol. p | same | diff | margin |
|---|---|---|---|
| 0.0 | 0.085 | 0.081 | 0.004 |
| 0.05 | 0.064 | 0.061 | 0.003 |
| 0.12 | 0.042 | 0.040 | 0.002 |
| 0.22 | 0.021 | 0.020 | 0.001 |

## Reproduce
```bash
for ds in orl yale; do
  python -m QuantumFace.benchmark base   $ds
  python -m QuantumFace.benchmark pure   $ds
  python -m QuantumFace.benchmark hybrid $ds
done
python -m QuantumFace.benchmark report
```
(SEED=42, deterministic.)
