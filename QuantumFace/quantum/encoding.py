"""
Phase 2 — quantum feature encoding.

Default: **angle encoding** (RY, one feature per qubit) — the FRQI/QPIXL-style choice
used by Zhu et al. 2024 and cheap in gate count (linear).

Documented optional alternatives (implemented, not the default):
  * amplitude encoding — packs 2**n features into n qubits, but needs normalisation
    and a costly state-preparation circuit;
  * basis encoding — one bit per qubit, only for pre-binarised features.

Also provides circuit-metrics logging (qubit count + depth) used by every experiment
so Phase 5 can report circuit depth automatically.
"""
from __future__ import annotations
import numpy as np
import pennylane as qml

from .. import config


# --------------------------------------------------------------------------
# encoders (act inside a qnode)
# --------------------------------------------------------------------------
def angle_encode(x, wires=None):
    """RY angle encoding: one feature per qubit. x already scaled to [0, pi]."""
    wires = wires if wires is not None else range(len(x))
    for i, w in enumerate(wires):
        qml.RY(x[i], wires=w)


def angle_encode_rx(x, wires=None):
    """RX variant (documented alternative axis)."""
    wires = wires if wires is not None else range(len(x))
    for i, w in enumerate(wires):
        qml.RX(x[i], wires=w)


def amplitude_encode(x, wires):
    """Amplitude encoding of 2**len(wires) features into len(wires) qubits."""
    n = len(wires)
    vec = np.zeros(2 ** n, dtype=float)
    vec[: min(len(x), 2 ** n)] = x[: 2 ** n]
    norm = np.linalg.norm(vec)
    vec = vec / norm if norm > 0 else vec
    qml.AmplitudeEmbedding(vec, wires=wires, normalize=True, pad_with=0.0)


def basis_encode(bits, wires):
    """Basis encoding: bits in {0,1} -> computational-basis state."""
    qml.BasisEmbedding(np.asarray(bits).astype(int), wires=wires)


ENCODERS = {"angle": angle_encode, "angle_rx": angle_encode_rx,
            "amplitude": amplitude_encode, "basis": basis_encode}


# --------------------------------------------------------------------------
# circuit metrics logger (qubits + depth)  — used across Phases 3-5
# --------------------------------------------------------------------------
def circuit_metrics(qnode, *args, **kwargs) -> dict:
    """Return {'n_qubits', 'depth', 'n_gates'} for a constructed qnode call."""
    specs = qml.specs(qnode)(*args, **kwargs)
    res = specs.get("resources", specs)
    depth = getattr(res, "depth", None)
    n_gates = getattr(res, "num_gates", None)
    n_wires = getattr(res, "num_wires", None)
    if depth is None and isinstance(specs, dict):  # older PennyLane dict form
        depth = specs.get("depth")
        n_gates = specs.get("num_operations") or specs.get("gate_sizes")
        n_wires = specs.get("num_used_wires")
    return {"n_qubits": int(n_wires) if n_wires else None,
            "depth": int(depth) if depth else None,
            "n_gates": int(n_gates) if isinstance(n_gates, int) else None}


def demo_encoding_circuit(n_qubits=config.N_QUBITS, encoding=config.ENCODING):
    dev = qml.device(config.SIM_DEVICE, wires=n_qubits)

    @qml.qnode(dev)
    def circ(x):
        ENCODERS[encoding](x, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circ


if __name__ == "__main__":
    circ = demo_encoding_circuit()
    x = np.linspace(0, np.pi, config.N_QUBITS)
    out = circ(x)
    print("angle-encoded <Z>:", np.round(np.asarray(out), 3))
    print("circuit metrics:", circuit_metrics(circ, x))
