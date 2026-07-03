"""Tests for the backend selector, resumable training, and dashboard wiring."""
import numpy as np
import pytest


# --- quantum backend selector ---------------------------------------------
def test_sim_backends_execute_here():
    from QuantumFace.quantum import backends
    for kind in ("sim", "noise"):
        info = backends.describe_backend(kind, wires=4)
        assert info["executes_here"] is True
        assert info["simulator_only_default"] is True


def test_ibm_backend_is_gated_and_offline_safe():
    from QuantumFace.quantum import backends
    info = backends.describe_backend("ibm", wires=4)
    assert info["executes_here"] is False          # never runs in-repo
    assert "experimental" in info["note"]
    # requesting the device without token/opt-in must refuse, not silently proceed
    with pytest.raises(RuntimeError):
        backends.get_device("ibm", wires=4)


# --- resumable / checkpointed training ------------------------------------
def test_resumable_training_reaches_target(tmp_path):
    from QuantumFace.quantum import train as T
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(0, np.pi, (8, 8)).astype(np.float32)
    ytr = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    ck = tmp_path / "ck.pkl"
    r1 = T.train_resumable(Xtr, ytr, 8, 4, "pure", total_steps=4, ckpt_path=ck, chunk=2)
    assert r1["done"] is False and r1["step"] == 2
    r2 = T.train_resumable(Xtr, ytr, 8, 4, "pure", total_steps=4, ckpt_path=ck, chunk=2)
    assert r2["done"] is True and r2["step"] == 4
    model = T.model_from_ckpt(ck)
    pred, _ = T.predict(model, Xtr, 4)
    assert pred.shape == (8,) and model["n_params"] == 52


# --- dashboard wiring (no live server) ------------------------------------
def test_dashboard_module_and_service_wire_up():
    import streamlit  # noqa: F401  - ensures the dep is installed
    from QuantumFace.frontend import dashboard
    assert callable(dashboard.main)
    # the service the dashboard wraps must construct + work (covered deeper in API tests)
    from QuantumFace.api.service import RecognitionService
    svc = RecognitionService(db_path=":memory:", secure=True)
    assert svc.users() == []
