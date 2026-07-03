"""
Quantum device/backend selector.

DEFAULT AND EVERYWHERE IN THIS REPO: local simulators. Real quantum hardware is an
OPT-IN, EXPERIMENTAL path that requires the user's own IBM Quantum account/token and is
NEVER assumed to be faster or more accurate than simulation (NISQ devices are noisy).

    get_device("sim",   wires=8)            # default.qubit state-vector simulator
    get_device("noise", wires=8)            # default.mixed density-matrix (for noise)
    get_device("ibm",   wires=8, backend="ibm_brisbane")  # experimental real hardware

The "ibm" path needs `pennylane-qiskit` + `qiskit-ibm-runtime` installed and an IBM
token in the environment (QF_IBM_TOKEN or the saved Qiskit account). It is validated by
`describe_backend` without submitting a job, so the selection logic is unit-testable
offline. Submitting real circuits is left to the user, who owns the quota and consent.
"""
from __future__ import annotations
import os
import pennylane as qml

from .. import config


def ibm_available() -> bool:
    """True only if the IBM plugin stack AND a token are present. No network call."""
    try:
        import pennylane_qiskit  # noqa: F401
    except Exception:
        return False
    return bool(os.environ.get("QF_IBM_TOKEN"))


def get_device(kind: str = "sim", wires: int = config.N_QUBITS, shots=None,
               backend: str = "ibm_brisbane"):
    """Return a PennyLane device. `kind`: 'sim' | 'noise' | 'ibm'."""
    if kind == "sim":
        return qml.device(config.SIM_DEVICE, wires=wires)
    if kind == "noise":
        return qml.device(config.NOISE_DEVICE, wires=wires)
    if kind == "ibm":
        if config.USE_REAL_QPU is False and os.environ.get("QF_ALLOW_QPU") != "1":
            raise RuntimeError(
                "Real-QPU path is disabled by default. This is experimental, consumes "
                "your IBM Quantum quota, and NISQ noise may worsen results. To enable, "
                "set QF_ALLOW_QPU=1 and provide QF_IBM_TOKEN, and install "
                "pennylane-qiskit + qiskit-ibm-runtime.")
        if not ibm_available():
            raise RuntimeError(
                "IBM backend requested but pennylane-qiskit and/or QF_IBM_TOKEN are "
                "missing. pip install pennylane-qiskit qiskit-ibm-runtime and export "
                "QF_IBM_TOKEN=<your token>.")
        return qml.device("qiskit.remote", wires=wires, backend=backend,
                          shots=shots or 1024, token=os.environ["QF_IBM_TOKEN"])
    raise ValueError(f"unknown backend kind {kind!r}; use 'sim' | 'noise' | 'ibm'")


def describe_backend(kind: str = "sim", wires: int = config.N_QUBITS) -> dict:
    """Report backend selection WITHOUT submitting a job (safe/offline for sim/noise)."""
    info = {"kind": kind, "wires": wires, "simulator_only_default": not config.USE_REAL_QPU}
    if kind in ("sim", "noise"):
        dev = get_device(kind, wires)
        info["device_name"] = dev.name
        info["executes_here"] = True
    else:  # ibm
        info["device_name"] = "qiskit.remote (IBM Quantum)"
        info["executes_here"] = False
        info["ibm_plugin_installed"] = _plugin_installed()
        info["token_present"] = bool(os.environ.get("QF_IBM_TOKEN"))
        info["enabled"] = ibm_available() and (
            config.USE_REAL_QPU or os.environ.get("QF_ALLOW_QPU") == "1")
        info["note"] = ("experimental; requires user IBM token + QF_ALLOW_QPU=1; "
                        "NISQ noise not assumed to beat simulation")
    return info


def _plugin_installed() -> bool:
    try:
        import pennylane_qiskit  # noqa: F401
        return True
    except Exception:
        return False


if __name__ == "__main__":
    for k in ("sim", "noise", "ibm"):
        print(describe_backend(k))
