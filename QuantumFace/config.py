"""
QuantumFace — central configuration.

Every tunable knob lives here so experiments are reproducible and a single
place documents the (small) scale of this research prototype.

THIS IS A RESEARCH / EDUCATION CODEBASE, NOT A PRODUCTION BIOMETRIC SYSTEM.
All quantum circuits run on local *simulators* by default.
"""
from __future__ import annotations
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
REPO_DIR = PKG_DIR.parent
# Datasets live inside the package so QuantumFace is self-contained.
DATA_DIR = PKG_DIR / "dataset"                    # local datasets + LFW cache
ORL_DIR = DATA_DIR / "att_faces"                  # ORL / AT&T (research use)
YALE_DIR = DATA_DIR / "yalefaces"                 # Yale Faces (research use)
RESULTS_DIR = REPO_DIR / "results"
DOCS_DIR = PKG_DIR / "docs"
DB_PATH = PKG_DIR / "database" / "quantumface.db"
KEY_DIR = PKG_DIR / "security" / "keys"

for _d in (DATA_DIR, RESULTS_DIR, DOCS_DIR, DB_PATH.parent, KEY_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Classical baseline (Phase 1)
# ---------------------------------------------------------------------------
FACE_SIZE = 160                 # FaceNet InceptionResnetV1 expects 160x160
EMBED_DIM = 512                 # FaceNet embedding dimensionality
FACENET_PRETRAINED = "vggface2" # pretrained weights tag (downloaded on first use)
CLASSICAL_CLF = "knn"           # 'knn' or 'svm' for closed-set recognition

# ---------------------------------------------------------------------------
# Dimensionality reduction + quantum encoding (Phase 2)
# ---------------------------------------------------------------------------
N_QUBITS = int(os.environ.get("QF_N_QUBITS", 8))   # default 8, expandable to 16
PCA_COMPONENTS = N_QUBITS                            # angle encoding: 1 feature/qubit
ENCODING = "angle"              # 'angle' (default) | 'amplitude' | 'basis'

# ---------------------------------------------------------------------------
# QCNN / training (Phases 3-4)
# ---------------------------------------------------------------------------
N_CONV_LAYERS = 2
TRAIN_STEPS = int(os.environ.get("QF_STEPS", 80))
LEARNING_RATE = 0.05
N_SUBJECTS = int(os.environ.get("QF_SUBJECTS", 4))  # closed-set identities

# ---------------------------------------------------------------------------
# Quantum backend (simulators only by default)
# ---------------------------------------------------------------------------
SIM_DEVICE = "default.qubit"    # state-vector simulator (PennyLane)
NOISE_DEVICE = "default.mixed"  # density-matrix sim for depolarising noise
USE_REAL_QPU = False            # NEVER assume hardware; opt-in, experimental only
NOISE_LEVELS = [0.0, 0.05, 0.12, 0.22]

# ---------------------------------------------------------------------------
# Security (Phase 6) — Kyber KEM protects stored embeddings AT REST only.
# ---------------------------------------------------------------------------
KYBER_VARIANT = "ML-KEM-512"    # via pure-python kyber-py (FIPS 203)
