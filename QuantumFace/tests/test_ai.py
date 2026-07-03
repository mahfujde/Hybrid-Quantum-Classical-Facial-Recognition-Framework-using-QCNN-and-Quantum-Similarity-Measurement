"""Tests for the optional AI-features modules (all offline)."""
import numpy as np
import pytest

from QuantumFace.ai import antispoof, emotion, xai, continual


# --- anti-spoofing / liveness ---------------------------------------------
def test_antispoof_separates_simulated_spoof():
    from QuantumFace.utils import data
    X, _, _ = data.load_dataset("orl", n_subjects=5)
    r = antispoof.evaluate_detector(X)
    assert r["genuine_mean"] > r["spoof_mean"]      # genuine sharper than print/replay
    assert r["best_accuracy"] >= 0.7


def test_liveness_needs_sequence():
    single = np.random.rand(1, 20, 20).astype(np.float32)
    assert antispoof.blink_liveness(single)["live"] is None   # can't prove from 1 frame
    frames = np.random.rand(4, 20, 20).astype(np.float32)     # varying -> motion
    out = antispoof.blink_liveness(frames)
    assert out["live"] in (True, False) and out["n_frames"] == 4


# --- emotion (Yale expressions) -------------------------------------------
def test_emotion_labels_and_run():
    X, y, names = emotion.load_yale_expressions()
    assert names == emotion.EMOTIONS and X.shape[0] == len(y)
    r = emotion.train_emotion_classifier()
    assert r["n_classes"] == 6 and 0.0 <= r["accuracy"] <= 1.0
    assert len(r["confusion_matrix"]) == 6


def test_age_mask_are_honest_stubs():
    assert emotion.AgeEstimator().available is False
    assert emotion.MaskDetector().available is False
    with pytest.raises(NotImplementedError):
        emotion.AgeEstimator().estimate(np.zeros((10, 10)))


# --- XAI -------------------------------------------------------------------
def test_occlusion_saliency_shape():
    from QuantumFace.classical.embed import EigenfaceEmbedder
    from QuantumFace.utils import data
    X, _, _ = data.load_dataset("orl", n_subjects=3)
    emb = EigenfaceEmbedder(dim=16).fit(X)
    heat = xai.occlusion_saliency(X[1], X[0], emb, patch=16, stride=16)
    assert heat.shape == X[1].shape
    assert 0.0 <= heat.min() and heat.max() <= 1.0


def test_quantum_feature_attribution():
    from QuantumFace.quantum import train as T
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(0, np.pi, (8, 8)).astype(np.float32)
    ytr = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    model = T.train_pure(Xtr, ytr, 8, 4, steps=2)
    r = xai.quantum_feature_attribution(model, Xtr, ytr, 4, n_repeats=1)
    assert r["n_features"] == 8 and 0 <= r["most_important_qubit"] < 8
    assert len(r["feature_importance"]) == 8


# --- continual learning ----------------------------------------------------
def test_prototype_incremental_enrolment():
    clf = continual.PrototypeClassifier()
    a = np.tile([1.0, 0.0, 0.0], (4, 1)).astype(np.float32)
    b = np.tile([0.0, 1.0, 0.0], (4, 1)).astype(np.float32)
    clf.enroll(0, a)
    assert clf.n_classes == 1
    clf.enroll(1, b)                                  # O(1) add, doesn't touch class 0
    assert clf.n_classes == 2
    assert clf.predict(a[:1])[0] == 0 and clf.predict(b[:1])[0] == 1


def test_no_catastrophic_forgetting():
    r = continual.measure_forgetting(dataset="orl", n_subjects=6, batch=2)
    assert r["forgetting"] <= 0.05                    # NCM -> ~zero forgetting by design
    assert r["trajectory"][-1]["classes_seen"] == 6
