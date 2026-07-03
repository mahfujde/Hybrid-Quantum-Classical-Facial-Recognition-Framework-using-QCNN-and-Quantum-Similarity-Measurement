"""
Phase 6 — Streamlit dashboard.

Run:  streamlit run QuantumFace/frontend/dashboard.py

Tabs:
  * Register / Recognize — upload a face (or use a local webcam snapshot, opt-in and
    kept local), see the detected face, the cosine similarity, the match result, and
    the measured end-to-end latency.
  * Quantum circuit — a rendered QCNN circuit diagram + logged depth/qubit count.

RESEARCH/EDUCATION prototype. No images are uploaded anywhere; everything is local.
"""
from __future__ import annotations
import io
import sys
import time
from pathlib import Path

import numpy as np

try:
    import streamlit as st
except Exception:  # pragma: no cover
    raise SystemExit("streamlit not installed: pip install streamlit")

# Work both as a package module (`python -m`) and as a script (`streamlit run`).
# `streamlit run` executes this file with no package context, so add the repo root to
# sys.path and use absolute imports.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from QuantumFace import config
from QuantumFace.api.service import RecognitionService


@st.cache_resource
def _service():
    return RecognitionService()


def _img_bytes(uploaded) -> bytes:
    return uploaded.getvalue()


def main():
    st.set_page_config(page_title="QuantumFace", layout="wide")
    st.title("QuantumFace — hybrid quantum-classical face recognition")
    st.caption("RESEARCH / EDUCATION prototype. Quantum results are from simulators "
               "only. Not a production biometric system. Images stay local.")
    svc = _service()

    tab1, tab2, tab3 = st.tabs(["Register", "Recognize", "Quantum circuit"])

    with tab1:
        st.subheader("Register a face")
        name = st.text_input("Name")
        up = st.file_uploader("Face image", type=["png", "jpg", "jpeg", "pgm", "gif"],
                              key="reg")
        if up:
            st.image(up, width=160, caption="uploaded")
        if st.button("Register", disabled=not (name and up)):
            r = svc.register(name, _img_bytes(up))
            st.success(f"Registered #{r['id']} ({r['name']}) — "
                       f"embedder={r['embedder']}, encrypted={r['encrypted']}")

    with tab2:
        st.subheader("Recognize a face")
        thr = st.slider("Match threshold (cosine)", 0.0, 1.0, 0.5, 0.05)
        up = st.file_uploader("Probe image", type=["png", "jpg", "jpeg", "pgm", "gif"],
                              key="rec")
        if up:
            st.image(up, width=160, caption="probe")
        if st.button("Recognize", disabled=not up):
            t0 = time.perf_counter()
            r = svc.recognize(_img_bytes(up), threshold=thr)
            latency_ms = (time.perf_counter() - t0) * 1000
            if r.get("matched"):
                st.success(f"MATCH: {r['name']} (sim={r['similarity']:.3f})")
            else:
                st.warning(f"No match (best sim={r.get('similarity', float('nan')):.3f})")
            st.metric("Measured latency", f"{latency_ms:.1f} ms")
        st.divider()
        st.write("Registered users:", svc.users())

    with tab3:
        st.subheader("QCNN circuit")
        from QuantumFace.quantum import qcnn as Q
        desc = Q.describe_circuit(config.N_QUBITS)
        c1, c2, c3 = st.columns(3)
        c1.metric("Qubits", desc["n_qubits"])
        c2.metric("Circuit depth", desc["depth"])
        c3.metric("Variational params", desc["n_variational_params"])
        try:
            import pennylane as qml
            circ = Q.build_qcnn(config.N_QUBITS, config.N_SUBJECTS)
            conv, pool = Q.init_conv_pool(config.N_QUBITS)
            x = np.linspace(0, np.pi, config.N_QUBITS)
            fig, _ = qml.draw_mpl(circ)(x, conv, pool)
            st.pyplot(fig)
        except Exception as e:  # pragma: no cover
            st.info(f"Circuit diagram unavailable: {e}")


if __name__ == "__main__":
    main()
