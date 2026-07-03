# QuantumFace — Limitations, Dataset Licensing & Security Scope

This document consolidates the honest caveats a reviewer or user must read before
interpreting any number in this repository.

## Scale & simulation limitations

- **Tiny qubit registers.** The QCNN uses **8 qubits** (expandable to 16) against
  embeddings that are natively 512-dimensional (FaceNet) or 27–30-dimensional
  (offline eigenface fallback). Angle encoding maps one feature to one qubit, so most
  of the representation is discarded before the quantum step. Phase 2 measures this:
  on ORL, PCA 27→8 retains only ~30% of variance and drops classical k-NN accuracy
  from 0.833 to 0.333 — **before any quantum processing**. Attribute hybrid-model
  weakness to this reduction first, not to the circuit.
- **Simulation only.** All results come from PennyLane `default.qubit` (state-vector)
  and `default.mixed` (density-matrix) simulators. They do **not** reflect real NISQ
  hardware noise except in the explicit depolarising-noise SWAP sweep, which is
  labelled as such. No number here was produced on a physical quantum computer.
- **Small datasets & few identities.** Benchmarks use 4 identities and a few dozen
  images. This is a *demonstration that the pipeline behaves as expected*, not a
  result competitive with production face recognition (which trains on millions of
  images). Absolute accuracies are not meaningful beyond this toy scale.
- **Short training.** The reported benchmark uses 30 optimisation steps for speed.
  Numbers are honest for that budget; longer training (`QF_STEPS=200`) will differ.

## Reproducibility caveats observed in the build environment

- **FaceNet pretrained weights could not be downloaded** in the build sandbox (the
  GitHub release-assets CDN and Hugging Face were network-blocked). The code path is
  implemented and shape-tested; the sandbox benchmark therefore used the **offline
  eigenface** embedder, and every artefact records `embedding_source`. Re-run on a
  networked machine to obtain the FaceNet-embedding numbers.
- **LFW** is fetched via scikit-learn and was likewise blocked in the sandbox; it is
  supported and will run where the download is permitted (`QF_DATASETS=orl,yale,lfw`).

## Dataset licensing & ethics

Only publicly-licensed research datasets are used, under their stated terms:

- **ORL / AT&T Database of Faces** — free for research use (AT&T Laboratories
  Cambridge). Stored locally under `code/datasets/att_faces`.
- **Yale Face Database** — free for non-commercial research use. Stored locally under
  `code/datasets/yalefaces`.
- **LFW (Labeled Faces in the Wild)** — University of Massachusetts, research use;
  fetched on demand via scikit-learn.

No images of real people are scraped or used outside these datasets. Any webcam or
own-photo demo is **opt-in, processed locally, and never uploaded**.

## Security scope (Kyber embedding encryption)

The Kyber (ML-KEM-512, FIPS 203) + AES-256-GCM envelope in `security/crypto.py`:

- **Protects:** offline theft of the embedding database. A stolen `quantumface.db`
  yields only post-quantum-wrapped ciphertext; raw embeddings cannot be recovered
  without the decapsulation (secret) key.
- **Does NOT protect:** model-inversion/template-reconstruction, presentation
  (spoofing) attacks, a compromised endpoint holding the secret key, or data in
  transit (use TLS). This is **one control**, not "unhackable identity".

The FastAPI service additionally leaves **authentication and rate limiting as explicit
TODOs** — they are noted in `api/main.py` and are out of scope for this research demo.

## One-line summary

QuantumFace is a faithful, honestly-measured *teaching and research* comparison of
classical vs hybrid-quantum face-recognition pipelines on simulators at small scale.
Every number traces to a script in this repo; nothing is projected.
