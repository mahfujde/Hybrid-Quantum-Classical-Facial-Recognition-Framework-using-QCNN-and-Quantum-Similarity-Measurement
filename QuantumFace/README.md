# QuantumFace

**A research & education framework for benchmarking hybrid quantum-classical
machine learning against classical baselines on face recognition.**

> ⚠️ **This is NOT a production biometric system.** It is a reproducible teaching
> and research codebase. Every quantum circuit runs on a **local simulator** by
> default. No claim here should be read as a deployable identity product.

---

## What this is

QuantumFace produces an **honest, reproducible comparison** of three pipelines on a
small public face dataset, using **real measured numbers** produced by the scripts in
this repo — never projected or assumed:

1. **Classical CNN baseline** — pretrained FaceNet (InceptionResnetV1) → 512-dim
   embedding → cosine similarity + a k-NN/SVM closed-set classifier.
2. **Hybrid QCNN** — FaceNet embedding → PCA to 8–16 dims → angle encoding → a small
   variational Quantum Convolutional Neural Network → classical read-out head.
3. **Pure QCNN** — the quantum layers do most of the classification work, minimal
   classical head.

It also implements a **SWAP-test quantum similarity** measure between two encoded
faces, and a **post-quantum-encrypted** (Kyber / ML-KEM) embedding store.

## Current Limitations (read this first)

- **Tiny qubit registers.** We use **8 qubits** (expandable to 16), while FaceNet
  embeddings are 512-dimensional. PCA to 8–16 dims **loses information**; Phase 2
  measures exactly how much, isolated from the quantum step.
- **Simulation only.** Results come from PennyLane's `default.qubit` (state-vector)
  and `default.mixed` (density-matrix) simulators. They **do not** reflect real
  NISQ-hardware noise unless a noise model is explicitly enabled (we include a
  depolarising-noise sweep, clearly labelled).
- **Small datasets.** We use small subsets (a few identities) of ORL/AT&T, Yale, and
  LFW. This is **not** competitive with production classical face recognition, which
  trains on millions of images.
- **Security scope.** The Kyber KEM protects the **stored embedding database at rest**
  against offline theft. It does **not** protect against model inversion, presentation
  (spoofing) attacks, or endpoint compromise. See `security/README` for the exact
  threat model.
- **No unverified performance claims.** Any number in the docs traces to a script in
  this repo and the exact command that produced it. Anything unrun is marked
  **"TBD — pending experiment."**

## Datasets & licensing

Only publicly-licensed research datasets are used, under their stated terms:
ORL/AT&T, Yale Faces, and a small LFW subset (fetched via scikit-learn). No images of
real people are scraped or used outside these sets. Any webcam/own-photo demo is local,
opt-in, and never uploaded.

## Repository layout

```
QuantumFace/
├── config.py            # every tunable knob + scale documentation
├── requirements.txt
├── tasks.py             # CI-friendly runner: baseline | benchmark | test | all
├── classical/           # Phase 1: detection, alignment, FaceNet embedding, classifier
├── quantum/             # Phases 2-3: PCA encoding, QCNN, SWAP test (auto-logs depth/qubits)
├── models/              # trained artefacts (classical/, quantum/)
├── database/            # Phase 6: SQLite schema + DAO
├── security/            # Phase 6: Kyber KEM + AES-GCM embedding encryption
├── api/                 # Phase 6: FastAPI register/recognize/users/delete
├── frontend/            # Phase 6: Streamlit dashboard
├── ai/                  # optional AI features: anti-spoof, emotion, XAI, continual
├── utils/               # dataset loaders (ORL/Yale/LFW)
├── tests/               # Phase 7: pytest unit + integration tests
└── docs/                # baseline_results.md, benchmark_report.md, limitations, ai_features.md
```

## Quick start

```bash
pip install -r QuantumFace/requirements.txt

# Phase 1 — classical FaceNet baseline (writes docs/baseline_results.md)
python -m QuantumFace.tasks baseline

# Phase 5 — three-way benchmark (writes docs/benchmark_report.md)
python -m QuantumFace.tasks benchmark

# Tests
python -m QuantumFace.tasks test
```

Set scale via environment variables: `QF_N_QUBITS`, `QF_SUBJECTS`, `QF_STEPS`.

## Reproducibility

Fixed `SEED = 42` throughout. Every results file records the environment, dataset
source, qubit count, and circuit depth. `docs/benchmark_report.md` lists the exact
command that reproduces each row of the results table.

## Measured results (this build)

Produced by `python -m QuantumFace.benchmark` on local simulators, 4 identities,
8 qubits, 30 training steps, `SEED=42`. **Embedding source = `eigenface`** because
FaceNet pretrained weights were network-blocked in the build sandbox — re-run on a
networked machine for the FaceNet numbers. Full detail in `docs/benchmark_report.md`.

| Dataset | Model | Acc | F1 | Train (s) | Params | Qubits | Depth |
|---|---|---|---|---|---|---|---|
| ORL | Classical (eigenface+kNN) | 0.833 | 0.831 | 0.00 | – | 0 | 0 |
| ORL | Hybrid QCNN | 0.667 | 0.631 | 19.3 | 72 | 8 | 20 |
| ORL | Pure QCNN | 0.333 | 0.250 | 19.3 | 52 | 8 | 20 |
| Yale | Classical (eigenface+kNN) | 0.214 | 0.196 | 0.00 | – | 0 | 0 |
| Yale | Hybrid QCNN | 0.143 | 0.106 | 21.0 | 72 | 8 | 20 |
| Yale | Pure QCNN | 0.357 | 0.326 | 21.2 | 52 | 8 | 20 |

These are toy-scale demonstration numbers, not competitive benchmarks — see
`docs/LIMITATIONS.md`. The SWAP-test similarity margin correctly decays as simulated
depolarising noise increases (ORL: 0.113 → 0.028 for p = 0 → 0.22).

## Optional AI features

`QuantumFace/ai/` adds four offline, honestly-measured extras (details + numbers in
`docs/ai_features.md`):

- **Anti-spoofing + liveness** — FFT/Laplacian presentation-attack cues; validated
  against *simulated* prints/replays (separation 0.25, AUC 1.0 vs simulated spoofs).
  Liveness requires a frame sequence and says so for a single image.
- **Emotion recognition** — trained on Yale's real expression labels; lands at ~chance
  (0.148 vs 0.167) because eigenface embeddings encode identity, not expression — a
  reported finding, not a bug. Age/mask are honest stubs (need offline-unavailable data).
- **Explainable AI** — occlusion saliency maps + permutation feature attribution over the
  QCNN's angle-encoded qubits.
- **Continual learning** — O(1) nearest-class-mean enrolment with **measured zero
  forgetting** as identities grow 2→8.

## Deployment (Docker + CI)

A CPU-only `Dockerfile` (repo root) builds a lean image and serves the API; `docker
compose up --build` runs the API (`:8000`, docs at `/docs`) and the Streamlit dashboard
(`:8501`) together. Tests/containers set `QF_FORCE_OFFLINE_EMBEDDER=1` so they run fully
offline (no FaceNet weight download).

```bash
docker build -t quantumface .
docker run --rm quantumface python -m pytest QuantumFace/tests -q
docker compose up --build          # API + dashboard
```

GitHub Actions (`.github/workflows/ci.yml`) runs the test suite on Python 3.10 & 3.11
(CPU-only PyTorch) on every push/PR, then builds the Docker image. Dependency versions
are pinned (`torch==2.2.2`, `numpy<2`, `opencv-python-headless==4.9.0.80`) so builds are
reproducible.

## Extended experiments

See `docs/extended_experiments.md` for measured results on: longer training (pure QCNN
0.333→0.417 at 200 steps; hybrid *overfits* 0.667→0.333), the 16-qubit run and the ~60×
simulation-cost wall it exposes, the gated experimental real-QPU backend
(`quantum/backends.py`), and the `scripts/regenerate_facenet_lfw.py` handoff for producing
real FaceNet + LFW numbers on a networked machine.

## Status

All phases (0–7), the optional AI features, and the extended experiments are implemented
and tested (31/31 passing). Quantum work is simulator-only by default; the real-QPU path
is experimental and opt-in. See `PROGRESS_MEMORY.md`
at the repo root for the live build log, `docs/benchmark_report.md` and
`docs/baseline_results.md` for measured results, `docs/reproducibility.md` for
reproduction steps, and `docs/LIMITATIONS.md` for the full caveats.
