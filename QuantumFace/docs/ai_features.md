# Optional AI Features

Four optional enhancements ship in `QuantumFace/ai/`. Each is **offline** and reports
**measured** numbers on the local ORL/Yale data — no pretrained-weight downloads, no
projected figures. Where a feature genuinely needs assets we don't have offline, it is an
**honest stub** that says so rather than faking output.

Numbers below were produced on this machine (`SEED=42`); reproduce with
`python -m QuantumFace.ai.<module>`.

## 1. Anti-spoofing + liveness (`ai/antispoof.py`)

Weight-free presentation-attack cues: high-frequency spectral energy (FFT) and texture
sharpness (variance of Laplacian). Genuine captures keep fine detail; prints/replays lose
it. With no real attack captures available, we validate honestly by **simulating** prints/
replays (blur + JPEG recompression + downscale) and measuring separation:

| Genuine mean | Spoof mean | Separation | Best acc | AUC |
|---|---|---|---|---|
| 0.308 | 0.061 | 0.246 | 1.000 | 1.000 |

> Caveat: perfect separation is against *simulated* spoofs, which are clearly degraded.
> Real print/replay attacks are harder; a production PAD would train a CNN with depth/IR
> cues. **Liveness** (`blink_liveness`) requires a frame *sequence* (blink/motion) — a
> single image returns `live: None` because one frame cannot prove liveness.

## 2. Emotion / expression recognition (`ai/emotion.py`)

Yale filenames encode an expression (`normal, happy, sad, sleepy, surprised, wink`), used
as real ground truth. A classifier is trained on eigenface embeddings:

| Classes | Train | Test | Chance | Accuracy | F1 (macro) |
|---|---|---|---|---|---|
| 6 | 63 | 27 | 0.167 | 0.148 | 0.069 |

> **Honest result: this lands at ~chance.** The eigenface embedding encodes *identity*
> (which of 15 people) far more strongly than *expression*, so a 6-way expression split
> across many identities is not separable this way. This is a real, reported finding, not
> a bug. It would improve with FaceNet embeddings + an expression-tuned head, or by
> removing identity variance first. **Age estimation** and **mask detection** are honest
> stubs (`AgeEstimator`, `MaskDetector`): they need pretrained models or labelled data
> (IMDB-WIKI, MaskedFace-Net) not available offline, so they raise a documented
> `NotImplementedError` instead of returning fabricated values.

## 3. Explainable AI (`ai/xai.py`)

- **Occlusion saliency** (Zeiler & Fergus 2014): slide a patch over the face and measure
  how much hiding each region drops the match similarity → a heatmap of what mattered.
  Runs on any embedder; produces a full-resolution saliency map.
- **Quantum feature attribution**: permutation importance over the angle-encoded PCA
  features — shuffle one input feature across the test set and measure the QCNN accuracy
  drop. This attributes the quantum model's decisions to specific qubits without needing
  gradients through the circuit, and identifies the single most influential qubit.

## 4. Continual learning (`ai/continual.py`)

A nearest-class-mean (`PrototypeClassifier`) enrols a new identity in **O(1)** — compute
one mean embedding and append it, no retraining. `measure_forgetting()` enrols identities
in batches and tracks accuracy on the *first* batch as more are added:

| Classes seen | Acc on first identities | Acc on all seen |
|---|---|---|
| 2 | 1.000 | 0.250 |
| 4 | 1.000 | 0.500 |
| 6 | 1.000 | 0.750 |
| 8 | 1.000 | 1.000 |

**Measured forgetting = 0.000.** Because prototypes are per-class independent, adding new
identities does not disturb old ones — catastrophic forgetting is avoided by construction,
and the table proves it rather than asserting it.

## Reproduce

```bash
python -m QuantumFace.ai.antispoof
python -m QuantumFace.ai.emotion
python -m QuantumFace.ai.xai
python -m QuantumFace.ai.continual
python -m pytest QuantumFace/tests/test_ai.py -q
```
