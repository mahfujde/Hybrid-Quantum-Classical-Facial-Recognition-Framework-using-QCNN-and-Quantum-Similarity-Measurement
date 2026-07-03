# Reproducing QuantumFace

Fixed `SEED = 42` throughout. A newcomer can clone the repo and reproduce the Phase 5
benchmark table with the commands below.

## 1. Environment

```bash
pip install -r QuantumFace/requirements.txt
```

Notes learned while building:
- Install **`torch==2.2.2`** + **`torchvision==0.17.2`** (pinned). Newer torch may pull
  large CUDA-only wheels on some platforms; the 2.2.2 CPU build has no such deps.
- Keep **`numpy<2`** (e.g. 1.26.4) and **`opencv-python-headless==4.9.0.80`** — this
  pairing is mutually compatible with torch 2.2.2.
- Kyber uses the pure-python **`kyber-py`** package (no C toolchain needed).

## 2. Classical baseline (Phase 1)

```bash
python -m QuantumFace.classical.baseline      # -> docs/baseline_results.md
```

Uses pretrained FaceNet if its weights can be downloaded, otherwise an offline
eigenface embedder; the source is recorded in the output.

## 3. Three-way benchmark (Phase 5)

Run in cached stages (each stage is short; quantum simulation is the slow part):

```bash
for ds in orl yale; do
  python -m QuantumFace.benchmark base   $ds
  python -m QuantumFace.benchmark pure   $ds
  python -m QuantumFace.benchmark hybrid $ds
done
python -m QuantumFace.benchmark report    # -> docs/benchmark_report.md, results/benchmark.json
```

Add LFW where the download is permitted: `QF_DATASETS=orl,yale,lfw`.
Scale knobs: `QF_N_QUBITS=8|16`, `QF_SUBJECTS=4`, `QF_STEPS=30|80|200`.

## 4. API + dashboard (Phase 6)

```bash
uvicorn QuantumFace.api.main:app --reload           # REST API
streamlit run QuantumFace/frontend/dashboard.py     # local dashboard
```

## 5. Tests (Phase 7)

```bash
python -m pytest QuantumFace/tests -q
```

19 unit + integration tests cover: config/loaders, embedding/alignment/verification,
PCA encoding, QCNN forward pass, SWAP test, training smoke, Kyber encryption
roundtrip + tamper detection, SQLite DAO, and the end-to-end register→recognize API
flow.
