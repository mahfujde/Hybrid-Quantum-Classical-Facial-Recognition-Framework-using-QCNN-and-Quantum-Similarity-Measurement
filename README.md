# QuantumFace

**A reproducible, honestly-benchmarked hybrid quantum-classical facial-recognition
research framework — QCNN + quantum similarity, compared head-to-head with a classical
baseline on local simulators.**

![CI](https://github.com/OWNER/quantumface/actions/workflows/ci.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue)
![status](https://img.shields.io/badge/status-research%20%2F%20education-orange)
![simulators](https://img.shields.io/badge/quantum-simulators%20only-6f42c1)

> ⚠️ **This is a research & education codebase, not a production biometric system.**
> Every quantum circuit runs on a **local simulator** by default, and **every number in
> this repo traces to a script you can re-run** — nothing is projected or marketed.

---

## What it does

QuantumFace produces a fair, three-way comparison on a small public face dataset:

1. **Classical baseline** — FaceNet (InceptionResnetV1) 512-d embedding → cosine
   similarity + a k-NN/SVM classifier.
2. **Hybrid QCNN** — embedding → PCA to 8–16 dims → angle encoding → a variational
   Quantum Convolutional Neural Network → a trainable classical head.
3. **Pure QCNN** — the quantum layers do the classifying, with a minimal classical head.

It also implements **SWAP-test quantum similarity** between two encoded faces, a
**post-quantum-encrypted** (Kyber / ML-KEM) embedding store, a **FastAPI** service, a
**Streamlit** dashboard, and four optional AI features (anti-spoofing, expression
recognition, explainable-AI saliency, continual learning).

## Architecture

```
image → detect (MTCNN/Haar) → align 160×160 → FaceNet embedding (512-d)
      → PCA (8–16) → RY angle encoding → QCNN [conv → pool → conv → measure]
      → readout (pure softmax | hybrid classical head) → recognition result
                                                       → SWAP-test similarity
      → Kyber+AES-GCM encrypted store (SQLite)  ·  FastAPI  ·  Streamlit dashboard
```

## Quickstart

```bash
pip install -r QuantumFace/requirements.txt

python -m QuantumFace.classical.baseline    # Phase 1: classical baseline
python -m QuantumFace.benchmark report      # Phase 5: three-way benchmark (see below)
python -m pytest QuantumFace/tests -q       # 31 unit + integration tests
uvicorn QuantumFace.api.main:app            # REST API (docs at /docs)
streamlit run QuantumFace/frontend/dashboard.py   # dashboard
```

Or with Docker:

```bash
docker compose up --build     # API on :8000, dashboard on :8501
```

## Measured results

Local simulators, ORL/Yale, 4 identities, 8 qubits, `SEED=42`. In this build the
**offline eigenface** embedder was used because the FaceNet weight CDN and LFW download
are blocked in the build sandbox — re-run `python -m QuantumFace.scripts.regenerate_facenet_lfw`
on a networked machine for real FaceNet + LFW numbers. Full detail in
`QuantumFace/docs/benchmark_report.md`.

| Dataset | Classical | Hybrid QCNN | Pure QCNN |
|---|---|---|---|
| ORL  | 0.833 | 0.667 | 0.333 |
| Yale | 0.214 | 0.143 | 0.357 |

Longer training (`docs/extended_experiments.md`): the pure QCNN improves to **0.417** at
200 steps, while the hybrid QCNN **overfits** (train loss → 0.225, test acc → 0.333) on
only 28 training images. The SWAP-test similarity margin correctly decays under simulated
depolarising noise (ORL: 0.113 → 0.028 for p = 0 → 0.22). Going 8 → 16 qubits slows a
gradient step ~60× — the exponential classical-simulation wall this project is built to
illustrate.

## Repository layout

```
QuantumFace/
├── classical/    detection, alignment, FaceNet/eigenface embedding, classifier
├── quantum/      PCA encoding, QCNN, SWAP test, training, backends (sim + gated QPU)
├── ai/           anti-spoofing, emotion, explainability, continual learning
├── database/     SQLite schema + DAO
├── security/     Kyber (ML-KEM) + AES-GCM embedding encryption
├── api/          FastAPI: /register /recognize /users /user/{id}
├── frontend/     Streamlit dashboard
├── experiments/  checkpointed long / 16-qubit runs
├── scripts/      FaceNet+LFW regeneration handoff
├── tests/        31 pytest unit + integration tests
├── docs/         baseline, benchmark, limitations, AI features, extended experiments
└── dataset/      ORL + Yale (local, research-licensed)
Dockerfile · docker-compose.yml · .github/workflows/ci.yml
```

## Honesty & limitations (please read)

- **Simulators only** by default; the real-QPU path is experimental, opt-in, and never
  assumed to beat simulation. Tiny qubit registers (8–16) vs 512-d embeddings mean PCA
  discards most information — quantified in Phase 2, isolated from the quantum step.
- **Small datasets, toy scale.** Absolute accuracies are not competitive with production
  face recognition and should not be read as such.
- **Security is scoped.** Kyber+AES-GCM protects the **stored embedding database at
  rest** against offline theft — *not* model inversion, spoofing, or endpoint
  compromise. API auth/rate-limiting are explicit TODOs.
- **No unverified claims.** Anything not yet run is marked "TBD". See
  `QuantumFace/docs/LIMITATIONS.md`.

## Datasets & licensing

ORL/AT&T and Yale (bundled, research use) and LFW (fetched on demand) — all public
research datasets used under their stated terms. No images of real people are scraped or
used outside these sets; any webcam/own-photo demo is local and opt-in.

## License

Released for research and educational use. Add a `LICENSE` file (e.g. MIT) before public
distribution; dataset terms remain governed by their respective providers.
