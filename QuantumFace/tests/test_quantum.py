"""Phases 2-4 tests: encoding, reduction, QCNN forward, SWAP test, training smoke."""
import numpy as np
import pytest

from QuantumFace import config
from QuantumFace.quantum import encoding, reduce as R, qcnn as Q, swap, train as T


def test_angle_encoding_metrics():
    circ = encoding.demo_encoding_circuit(n_qubits=8)
    x = np.linspace(0, np.pi, 8)
    out = np.asarray(circ(x))
    assert out.shape == (8,)
    m = encoding.circuit_metrics(circ, x)
    assert m["n_qubits"] == 8 and m["depth"] >= 1


def test_pca_reducer_range_and_var():
    rng = np.random.default_rng(0)
    emb = rng.random((30, 40)).astype(np.float32)
    red = R.PCAReducer(dim=8).fit(emb)
    z = red.transform(emb)
    assert z.shape == (30, 8)
    assert z.min() >= -1e-6 and z.max() <= np.pi + 1e-6
    assert 0.0 <= red.explained_variance_ratio <= 1.0


def test_qcnn_forward_and_metrics():
    circ = Q.build_qcnn(n_qubits=8, n_classes=4)
    conv, pool = Q.init_conv_pool(8)
    out = circ(np.linspace(0, np.pi, 8), conv, pool)
    assert len(out) == Q.n_survivors(8) == 4
    desc = Q.describe_circuit(8)
    assert desc["n_qubits"] == 8 and desc["n_variational_params"] == 52


def test_swap_test_identity_is_high():
    fid = swap.build_swap_test(n_feat=4)
    x = np.array([0.5, 1.0, 1.5, 2.0])
    assert fid(x, x) > 0.95                 # identical states -> fidelity ~1
    assert fid(x, np.pi - x) < fid(x, x)    # different states -> lower


def test_swap_noise_reduces_margin():
    feats = np.array([[0.2, 0.3, 0.4, 0.5]] * 3 + [[2.5, 2.6, 2.7, 2.8]] * 3)
    labels = np.array([0, 0, 0, 1, 1, 1])
    rep = swap.similarity_report(feats, labels, n_feat=4, noise_levels=[0.0, 0.2])
    assert rep["swap_margin"][0] >= rep["swap_margin"][1]  # noise shrinks margin


def test_train_pure_smoke():
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(0, np.pi, (8, 8)).astype(np.float32)
    ytr = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    m = T.train_pure(Xtr, ytr, 8, 4, steps=2)
    assert len(m["history"]) == 2 and m["n_params"] == 52
    pred, lat = T.predict(m, Xtr, 4)
    assert pred.shape == (8,) and lat >= 0
