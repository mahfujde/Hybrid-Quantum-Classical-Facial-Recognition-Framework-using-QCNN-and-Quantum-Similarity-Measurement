"""Phase 1 tests: embedding, alignment, verification, classifier (offline eigenface)."""
import numpy as np

from QuantumFace.classical import embed, detect, baseline


def test_eigenface_embedder_shape_and_determinism():
    rng = np.random.default_rng(0)
    X = rng.random((20, 32, 32), dtype=np.float32)
    e1 = embed.EigenfaceEmbedder(dim=8).fit(X).embed(X)
    e2 = embed.EigenfaceEmbedder(dim=8).fit(X).embed(X)
    assert e1.shape == (20, 8)
    np.testing.assert_allclose(e1, e2, atol=1e-6)  # deterministic under SEED


def test_align_prealigned_is_face_size():
    from QuantumFace import config
    out = detect.align_prealigned(np.random.rand(50, 40).astype(np.float32))
    assert out.shape == (config.FACE_SIZE, config.FACE_SIZE)
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_verification_metrics_separates_identities():
    # two well-separated clusters -> same-mean cos should exceed diff-mean cos
    a = np.tile([1.0, 0.0, 0.0], (5, 1)) + 0.01
    b = np.tile([0.0, 1.0, 0.0], (5, 1)) + 0.01
    emb = np.vstack([a, b]).astype(np.float32)
    y = np.array([0] * 5 + [1] * 5)
    m = baseline.verification_metrics(emb, y)
    assert m["verif_same_mean"] > m["verif_diff_mean"]


def test_baseline_run_orl_offline():
    r = baseline.run_dataset("orl", n_subjects=3, source="eigenface")
    assert 0.0 <= r["accuracy"] <= 1.0
    assert r["embedding_source"] == "eigenface"
    assert r["confusion_matrix"]
