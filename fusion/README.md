# Stage D — 3-sensor hybrid late fusion (Camera+LiDAR BEVFusion x Camera+Radar CRN)

No new network was trained. Two independent perception branches are merged at object level:

```
Camera+LiDAR BEVFusion --> pred_camera_lidar (nuScenes json, global frame) --\
                                                                              +--> Hungarian association --> Unified Object List --> MATLAB
Camera+Radar CRN       --> pred_camera_radar (nuScenes json, global frame) --/
```

`late_fusion.py` (all constants are priors fixed **before** any evaluation):
* candidates: boxes with score >= 0.05; association: Hungarian on BEV centre distance, **same class only**, class gate (car 2.5 m, truck 3.5, bus/trailer/construction 4.0, pedestrian/bicycle/barrier 1.5, motorcycle 2.0, cone 1.0);
* matched pair: centre = weighted mean (0.7 x score_CL, 0.3 x score_CR); size, yaw, attribute from Camera+LiDAR; velocity = score-weighted mean; score = noisy-OR `1-(1-sA)(1-sB)`;
* unmatched: kept, score x 0.85 (Camera+LiDAR only) or x 0.6 (Camera+Radar only).

## Official nuScenes detection metrics, mini VAL (81 samples, devkit `NuScenesEval`, `mini_val`)

| System | mAP | NDS | mATE | mAOE | mAVE | Notes |
|---|---|---|---|---|---|---|
| Camera+LiDAR, official pretrained | 0.570 | 0.576 | 0.411 | 0.497 | 0.421 | contaminated (trained on trainval) |
| Camera+Radar CRN R18, official pretrained | 0.433 | 0.476 | 0.575 | 0.640 | 0.425 | contaminated |
| **Fused: official CL + CRN** | 0.574 | 0.578 | 0.439 | 0.494 | 0.416 | both inputs contaminated |
| Camera+LiDAR, **our** trained (Stage B) | 0.192 | 0.207 | 0.622 | 1.136 | 0.987 | mini TRAIN only |
| **Fused: our CL + CRN** | 0.436 | 0.415 | 0.556 | 0.981 | 0.527 | CRN input is contaminated |

Ablations on the same 81 samples (run after the baseline; **not** used to pick the baseline): no penalty 0.557/0.569 (official) and 0.437/0.426 (ours); velocity from CL only 0.574/0.574 and 0.436/0.374; velocity from CR only 0.574/0.576 and 0.436/0.425; equal weights 0.566/0.572 and 0.436/0.416; strong penalty (0.7/0.4) 0.576/0.579 and 0.426/0.410. Full table: `notebooks/bev-real-fusion-eval/out/outputs/fusion_eval/fusion_eval.json`.

## What the numbers do and do not say

1. **No demonstrable AP/NDS gain over Camera+LiDAR alone.** +0.004 mAP / +0.002 NDS is inside the +/-0.003 run-to-run variation we measured for the same checkpoints. Centre error gets slightly worse (mATE 0.439 vs 0.411) because matched centres are pulled toward the less accurate Camera+Radar centres.
2. **Fusion mostly trades precision for recall.** At a fixed score threshold (`unified/operating_points.json`, precision/recall against GT in the ego frame, class-gated 1:1 matching):

   | System | thr 0.3 P / R | thr 0.5 P / R | thr 0.7 P / R |
   |---|---|---|---|
   | Camera+LiDAR official | 0.94 / 0.62 | 0.98 / 0.47 | 1.00 / 0.22 |
   | Camera+Radar CRN | 0.46 / 0.67 | 0.75 / 0.51 | 0.96 / 0.23 |
   | Fused (official CL + CRN) | 0.68 / 0.73 | 0.89 / 0.65 | 0.96 / 0.52 |

   At a comparable precision of about 0.9 the fused list recalls a little more (0.65 vs 0.62), but at precision >= 0.96 Camera+LiDAR alone is as good or better. A fixed 0.3 threshold on the fused list is **not** a good operating point (many low-quality Camera+Radar-only boxes); use 0.5 or higher.
3. **A weak branch cannot be rescued with fixed source priors.** With our (weak) Camera+LiDAR model the fused list has mAP ~ Camera+Radar alone (0.436 vs 0.433) but **lower NDS** (0.415 vs 0.476), because matched boxes inherit yaw/size/velocity from the weaker Camera+LiDAR side (mAOE 0.98 vs 0.64). Taking velocity from Camera+Radar helps (NDS 0.425) but not enough. A reliability-aware rule would be needed; it was not tuned here because there is no held-out split (81 samples).
4. **Contamination.** Every "official" row and the CRN input in every fused row were trained on nuScenes trainval, which contains the mini scenes. Treat them as a **pretrained integration reference**, not as evidence of generalization. Only "Camera+LiDAR, our trained" is free of mini-VAL exposure.
5. Mini VAL has 81 samples / 2 scenes and no trailer, construction_vehicle or barrier GT (mAP is capped at 0.70). Not a full-nuScenes benchmark.

## Unified Object List (`unified/`)

* `unified_pretrained_ego.csv`, `unified_pretrained.mat`: official Camera+LiDAR x CRN. `unified_ours_ego.csv`, `unified_ours.mat`: our Camera+LiDAR x CRN.
* Columns: `sample_idx, sample_token, class, x, y, z, w, l, h, yaw, vx, vy, score, source, score_cl, score_cr`. Ego frame: x forward, y left, z up, yaw CCW from +x; `sample_idx` is identical to the L3E exports. `source` = `both` (sensors agree) / `cl` / `cr`. All boxes are kept; filter by `score` (>= 0.5 recommended).
* `load_unified.m` (verified headless in MATLAB), `unified_bev_sample5.png`, `unified_gt_check.json`, `operating_points.json`.
* nuScenes-format fused submissions (global frame): `notebooks/bev-real-fusion-eval/out/outputs/fusion_eval/submissions/`.

Re-run: `python late_fusion.py --a <cl_results.json> --b <cr_results.json> --out fused.json`; evaluation notebook `bev-real-fusion-eval` (CPU, uses the frozen BEVFusion environment for the devkit).
