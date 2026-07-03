"""
Phase 3 — variational Quantum Convolutional Neural Network (QCNN).

Architecture (Cong, Choi & Lukin 2019; multi-gate variant per Zhu et al. 2024):

    angle encoding -> [multi-gate conv -> pooling] x L -> measurement -> readout

Two readout modes give the three-way comparison of Phase 5:
  * "pure"   : one qubit per class, softmax over Pauli-Z expectations
               (minimal classical head — the quantum layers do the classifying).
  * "hybrid" : a small TRAINABLE classical linear layer maps ALL measured qubit
               expectations to class logits (more classical work).

Circuit depth and qubit count are logged automatically for every built circuit.
"""
from __future__ import annotations
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .. import config
from .encoding import angle_encode, circuit_metrics


# --------------------------------------------------------------------------
# building blocks
# --------------------------------------------------------------------------
def multi_gate_conv(params, wires):
    """Trainable RX/RY/RZ per wire + CNOT ring. params: (len(wires), 3)."""
    for k, w in enumerate(wires):
        qml.RX(params[k, 0], wires=w)
        qml.RY(params[k, 1], wires=w)
        qml.RZ(params[k, 2], wires=w)
    for k in range(len(wires) - 1):
        qml.CNOT(wires=[wires[k], wires[k + 1]])
    if len(wires) > 2:
        qml.CNOT(wires=[wires[-1], wires[0]])


def pooling(params, wires):
    """Controlled-RZ pooling: halve the register, return surviving wires."""
    kept, discarded = wires[::2], wires[1::2]
    for k, (c, t) in enumerate(zip(discarded, kept)):
        qml.CRZ(params[k], wires=[c, t])
    return kept


# --------------------------------------------------------------------------
# QCNN circuit
# --------------------------------------------------------------------------
def build_qcnn(n_qubits=config.N_QUBITS, n_classes=config.N_SUBJECTS,
               noise=False, noise_p=0.0):
    """Return a qnode(x, conv, pool) -> list of Pauli-Z expectations on the
    surviving qubits (>= n_classes)."""
    dev = (qml.device(config.NOISE_DEVICE, wires=n_qubits) if noise
           else qml.device(config.SIM_DEVICE, wires=n_qubits))

    @qml.qnode(dev, interface="autograd")
    def circuit(x, conv, pool):
        angle_encode(x, wires=range(n_qubits))
        active = list(range(n_qubits))
        multi_gate_conv(conv[0], wires=active)
        if noise:
            for q in active:
                qml.DepolarizingChannel(noise_p, wires=q)
        active = pooling(pool, wires=active)                # n -> n/2
        multi_gate_conv(conv[1][:len(active)], wires=active)
        if noise:
            for q in active:
                qml.DepolarizingChannel(noise_p, wires=q)
        return [qml.expval(qml.PauliZ(q)) for q in active]

    return circuit


def init_conv_pool(n_qubits=config.N_QUBITS, seed=config.SEED):
    rng = np.random.default_rng(seed)
    conv = pnp.array(rng.uniform(0, np.pi, (config.N_CONV_LAYERS, n_qubits, 3)),
                     requires_grad=True)
    pool = pnp.array(rng.uniform(0, np.pi, (n_qubits // 2,)), requires_grad=True)
    return conv, pool


def n_survivors(n_qubits=config.N_QUBITS):
    return len(list(range(n_qubits))[::2])


# --------------------------------------------------------------------------
# readout heads
# --------------------------------------------------------------------------
TEMPERATURE = 3.0


def pure_readout(measurements, n_classes):
    """Softmax over the first n_classes qubit expectations (minimal classical head)."""
    vec = pnp.stack(measurements)[:n_classes] * TEMPERATURE
    ex = pnp.exp(vec - pnp.max(vec))
    return ex / pnp.sum(ex)


def hybrid_readout(measurements, weights, bias):
    """Trainable classical linear layer over ALL measurements -> softmax logits."""
    m = pnp.stack(measurements)
    logits = weights @ m + bias
    ex = pnp.exp(logits - pnp.max(logits))
    return ex / pnp.sum(ex)


def init_hybrid_head(n_measure, n_classes, seed=config.SEED):
    rng = np.random.default_rng(seed + 1)
    w = pnp.array(rng.normal(0, 0.5, (n_classes, n_measure)), requires_grad=True)
    b = pnp.array(np.zeros(n_classes), requires_grad=True)
    return w, b


def describe_circuit(n_qubits=config.N_QUBITS):
    """Return logged circuit metrics + parameter counts for a fresh QCNN."""
    circ = build_qcnn(n_qubits=n_qubits)
    conv, pool = init_conv_pool(n_qubits)
    x = np.linspace(0, np.pi, n_qubits)
    m = circuit_metrics(circ, x, conv, pool)
    m["n_variational_params"] = int(conv.size + pool.size)
    m["n_survivors"] = n_survivors(n_qubits)
    return m


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    print("QCNN circuit:", describe_circuit())
