"""Phase 0 smoke tests: package imports, config sanity, dataset loaders."""
import numpy as np

from QuantumFace import config
from QuantumFace.utils import data


def test_config_scale_is_small_and_simulator_only():
    assert config.N_QUBITS in (8, 16)
    assert config.PCA_COMPONENTS == config.N_QUBITS
    assert config.USE_REAL_QPU is False
    assert config.SIM_DEVICE == "default.qubit"


def test_load_orl_shapes():
    X, y, names = data.load_dataset("orl", n_subjects=3)
    assert X.ndim == 3 and X.shape[0] == len(y)
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert len(names) == 3
    assert set(np.unique(y)) == {0, 1, 2}


def test_load_yale_shapes():
    X, y, names = data.load_dataset("yale", n_subjects=3)
    assert X.shape[0] == len(y) > 0
    assert len(names) == 3
