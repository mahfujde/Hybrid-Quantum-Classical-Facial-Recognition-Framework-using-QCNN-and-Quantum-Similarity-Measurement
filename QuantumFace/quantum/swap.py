"""
Phase 3 — quantum similarity via the SWAP test.

The SWAP test (Buhrman et al. 2001) estimates the fidelity |<psi|phi>|^2 between the
angle-encoded states of two faces:

    <Z_ancilla> = P(0) - P(1) = |<psi|phi>|^2   (noiseless)

Gate noise biases the estimate toward 0, so the same/different-identity *margin*
shrinks as depolarising probability rises — reported as a noise-robustness sweep.
Runs on the state-vector simulator (noiseless) or the density-matrix simulator (noise).
"""
from __future__ import annotations
import numpy as np
import pennylane as qml

from .. import config


def build_swap_test(n_feat=4, noise=False, p=0.0):
    """SWAP test over n_feat features per state (2*n_feat + 1 wires)."""
    wires = list(range(2 * n_feat + 1))
    anc = wires[-1]
    a, b = wires[:n_feat], wires[n_feat:2 * n_feat]
    dev = (qml.device(config.NOISE_DEVICE, wires=wires) if noise
           else qml.device(config.SIM_DEVICE, wires=wires))

    @qml.qnode(dev)
    def swap(xa, xb):
        for i in range(n_feat):
            qml.RY(xa[i], wires=a[i])
            qml.RY(xb[i], wires=b[i])
        qml.Hadamard(wires=anc)
        for i in range(n_feat):
            qml.CSWAP(wires=[anc, a[i], b[i]])
            if noise:
                qml.DepolarizingChannel(p, wires=anc)
        qml.Hadamard(wires=anc)
        return qml.expval(qml.PauliZ(anc))

    def fidelity(xa, xb):
        return float(swap(np.asarray(xa)[:n_feat], np.asarray(xb)[:n_feat]))

    return fidelity


def similarity_report(feats, labels, n_feat=4, noise_levels=config.NOISE_LEVELS):
    """Mean SWAP fidelity for same- vs different-identity pairs, across noise levels.
    Returns margins (same - diff) that should shrink with noise."""
    feats, labels = np.asarray(feats), np.asarray(labels)
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    same_mask = labels[iu[0]] == labels[iu[1]]
    out = {"n_feat": n_feat, "noise_levels": list(noise_levels),
           "swap_same": [], "swap_diff": [], "swap_margin": []}
    for p in noise_levels:
        fid = build_swap_test(n_feat=n_feat, noise=(p > 0), p=p)
        sims = np.array([fid(feats[i], feats[j]) for i, j in zip(*iu)])
        same_m = float(sims[same_mask].mean()) if same_mask.any() else float("nan")
        diff_m = float(sims[~same_mask].mean()) if (~same_mask).any() else float("nan")
        out["swap_same"].append(same_m)
        out["swap_diff"].append(diff_m)
        out["swap_margin"].append(same_m - diff_m)
    return out


if __name__ == "__main__":
    import warnings; warnings.filterwarnings("ignore")
    fid = build_swap_test(n_feat=4)
    x = np.array([0.5, 1.0, 1.5, 2.0])
    print("fidelity(x,x)=%.3f (expect ~1)" % fid(x, x))
    print("fidelity(x,pi-x)=%.3f" % fid(x, np.pi - x))
