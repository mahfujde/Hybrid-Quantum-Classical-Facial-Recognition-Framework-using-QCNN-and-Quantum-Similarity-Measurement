# Phase 1 — Classical Baseline Results

> RESEARCH/EDUCATION prototype. Numbers below were produced by `python -m QuantumFace.classical.baseline` on the machine that generated this file. Every value traces to that run.

**Embedding source used:** `eigenface`  
- `facenet` = pretrained FaceNet InceptionResnetV1 (VGGFace2), 512-dim.
- `eigenface` = offline PCA-on-pixels fallback, used when the FaceNet pretrained weights could not be downloaded (GitHub release CDN blocked). **If you see `eigenface` here, the 'Classical CNN' framing does not apply — these are classical eigenface numbers; re-run on a networked machine for FaceNet.**

## Closed-set recognition + verification

| Dataset | Src | Dim | Clf | Acc | Prec | Rec | F1 | Verif AUC | same/diff cos |
|---|---|---|---|---|---|---|---|---|---|
| orl | eigenface | 27 | knn | 0.833 | 0.854 | 0.833 | 0.831 | 0.626 | 0.069/-0.043 |
| yale | eigenface | 30 | knn | 0.214 | 0.300 | 0.208 | 0.196 | 0.519 | 0.006/-0.026 |

Reproduce: `python -m QuantumFace.classical.baseline`
(config: N_SUBJECTS=4, SEED=42)
