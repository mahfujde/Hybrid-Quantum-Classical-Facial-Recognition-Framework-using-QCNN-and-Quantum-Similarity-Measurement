# Extended Experiments — longer training, 16 qubits, real-QPU path, FaceNet/LFW handoff

All numbers below were measured on the local simulator (`SEED=42`, ORL, 4 identities,
eigenface embeddings — FaceNet weights are network-blocked in this environment). Nothing
is projected.

## 1. Longer training (200 steps vs 30)

Run in checkpointed chunks (`QuantumFace/experiments/long_run.py`) because a full run
exceeds a single short session; Adam state is pickled so momentum survives resumes.

| Model | Steps | Loss start → end | Test acc |
|---|---|---|---|
| Pure QCNN (8q)   | 30  | 1.38 → ~1.05 | 0.333 |
| Pure QCNN (8q)   | 200 | 1.376 → 0.952 | **0.417** |
| Hybrid QCNN (8q) | 30  | 1.40 → ~0.60 | 0.667 |
| Hybrid QCNN (8q) | 200 | 1.397 → 0.225 | **0.333** |

**Findings (honest):**
- The **pure** QCNN improves with more training (0.333 → 0.417) — it was underfitting at
  30 steps.
- The **hybrid** QCNN *overfits*: its training loss collapses to 0.225 but test accuracy
  drops from 0.667 to 0.333. With only 28 training images and a 72-parameter model, long
  training memorises the training set. This is expected at this toy scale and is exactly
  why the limitations section warns against reading absolute accuracies as meaningful.

Reproduce:
```bash
for i in 1 2 3 4; do python -m QuantumFace.experiments.long_run step orl pure   8 200; done
python -m QuantumFace.experiments.long_run finalize orl pure   8 200
for i in 1 2 3 4; do python -m QuantumFace.experiments.long_run step orl hybrid 8 200; done
python -m QuantumFace.experiments.long_run finalize orl hybrid 8 200
```

## 2. 16-qubit run — and the simulation-cost wall

| Qubits | Circuit depth | Params | Cost per gradient step (28-sample set) |
|---|---|---|---|
| 8  | 20 | 52  | ~0.66 s |
| 16 | 32 | 104 | **> 40 s** |

Going from 8 to 16 qubits makes the state vector grow from 2^8 = 256 to 2^16 = 65 536
complex amplitudes, and a full-training-set gradient step jumps from sub-second to over
40 seconds — a ~60× slowdown. This is the **exponential classical-simulation cost** that
motivates the small qubit counts used throughout QuantumFace.

A minimal 16-qubit run (8 training images, 20 steps) completes and trains correctly
(loss 1.422 → 0.931, acc 0.250 ≈ chance on this tiny subset), confirming the pipeline
scales to 16 qubits functionally even though full-scale 16-qubit training is impractical
on a laptop simulator. Recorded in `results/long_run.json`.

## 3. Experimental real-QPU path (`quantum/backends.py`)

`get_device(kind, wires, backend)` selects the execution backend:

- `"sim"` → `default.qubit` (state vector) — the default everywhere in this repo.
- `"noise"` → `default.mixed` (density matrix) — for the depolarising-noise experiments.
- `"ibm"` → real IBM Quantum hardware — **experimental, opt-in, and disabled by default.**

The IBM path is gated three ways and will refuse to run unless all are satisfied:
`QF_ALLOW_QPU=1`, a token in `QF_IBM_TOKEN`, and `pennylane-qiskit` installed. It is
never assumed to be faster or more accurate than simulation — NISQ devices are noisy.
`describe_backend()` reports the selection without submitting a job, so the logic is
unit-tested offline. Submitting real circuits (and the quota/consent that implies) is
left to you.

```bash
pip install pennylane-qiskit qiskit-ibm-runtime
export QF_IBM_TOKEN=<your token>  QF_ALLOW_QPU=1
```

## 4. FaceNet + LFW handoff (`scripts/regenerate_facenet_lfw.py`)

The build environment blocks the FaceNet weight CDN, Hugging Face, and scikit-learn's LFW
download, so the in-repo numbers use the eigenface fallback. On any networked machine:

```bash
pip install -r QuantumFace/requirements.txt
python -m QuantumFace.scripts.regenerate_facenet_lfw            # ORL + Yale + LFW
```

It forces the real FaceNet embedder (erroring loudly rather than falling back), then
re-runs the Phase 1 baseline and Phase 5 benchmark, overwriting `docs/baseline_results.md`,
`docs/benchmark_report.md`, and `results/*.json` with real FaceNet + LFW numbers.
