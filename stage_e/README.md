# Stage E — Classical vs Learned on the same data (nuScenes-mini VAL, 81 samples)

**Classical** = the project's own Level-1 MATLAB code, called unmodified: `processLidarScan` (ROI, ego-body removal, RANSAC ground), `buildBevOccupancy`, `extractLidarDetections` (distance clustering, oriented boxes), `rasterizeRadarBev` (5 radars). **Learned** = the pretrained and own-trained models from Stages A-D. Same 81 keyframes, same GT, same criteria.

## Two things to know before reading any number

1. **Toolboxes (updated 25/09/2026).** The first run of this stage happened while MATLAB R2026a had only MATLAB, Simulink and the MCP toolbox, so four base-MATLAB shims (`matlab/shims/`: `pointCloud`, `pcfitplane`, `pcsegdist`, `objectDetection`) stood in for the Computer Vision / Lidar / Sensor Fusion functions. The missing toolboxes have since been installed, and the whole classical run was **repeated with the real toolbox functions**; every number below is the toolbox number. The shim run is kept for comparison (`classical_out_shim/`, `results/stage_e_metrics_shim.json`): F1 0.460 (shim) vs 0.459 (toolbox), occupancy object-cell recall identical to three decimals, per-frame occupied cells 975 vs 991, so the shims were faithful. `run_classical_nuscenes` now auto-detects: it adds the shims only if a toolbox function is missing. Project code is untouched; nothing under `cam_lidar_radar_ws` was modified.
2. **The classical thresholds were tuned for the Unreal scenario and were NOT retuned** (per the project's own rule: no tuning after seeing results). Only physical rig parameters changed, fixed before the run: BEV grid +/-50 m at 0.5 m, real nuScenes LiDAR/radar extrinsics, nuScenes ego body box, LiDAR horizontal FOV 360 deg. The classical result is therefore "the existing pipeline dropped onto real data", not the best a tuned classical pipeline could do. Camera classical detection is not applicable (Level 1's camera detector is an Unreal semantic oracle), so no classical camera row exists.

## Protocol (fixed before the first evaluation)

* GT evaluated: annotated objects with `num_lidar_pts >= 5` inside the 100x100 m BEV -> 2376 of 4441 boxes; other GT are "don't care" (a prediction that matches only a don't-care GT is ignored, not a false positive).
* Matching: Hungarian on ego-BEV centre distance, class-agnostic, per-GT-class gate (car 2.5 m, truck 3.5, bus/trailer/construction 4.0, pedestrian 1.5, motorcycle 2.0, bicycle 1.5, cone 1.0, barrier 1.5).
* Classical has no confidence, so it is a single operating point. Learned lists are thresholded at score >= 0.5 (0.3 and 0.7 also given).
* Occupancy: 200x200 cells (0.5 m), 1-cell tolerance (the tolerance of the project's own occupancy contract).

## E1. Object detection (class-agnostic, per image = mean objects per frame; GT = 29.3 per frame)

| System | Precision | Recall | F1 | Objects / img | Vehicle recall | Pedestrian recall | Centre error (m) |
|---|---|---|---|---|---|---|---|
| **Classical** project LiDAR pipeline (real toolboxes) | 0.43 | 0.50 | 0.46 | 35.5 | 0.63 | 0.18 | 0.89 |
| Learned official LiDAR-only (contaminated) | 0.99 | 0.73 | 0.84 | 23.5 | 0.74 | 0.71 | 0.15 |
| Learned official Camera+LiDAR (contaminated) | 0.99 | 0.79 | 0.88 | 26.3 | 0.79 | 0.78 | 0.16 |
| Learned Camera+Radar CRN (contaminated) | 0.74 | 0.78 | 0.76 | 37.4 | 0.75 | 0.86 | 0.34 |
| Learned fused CL+CR, score >= 0.5 (contaminated) | 0.87 | 0.93 | 0.90 | 39.9 | 0.92 | 0.97 | 0.19 |
| **Ours** LiDAR-only (mini-trained) | 1.00 | 0.15 | 0.26 | 4.4 | 0.19 | 0.05 | 0.23 |
| **Ours** Camera+LiDAR (mini-trained) | 1.00 | 0.19 | 0.32 | 5.7 | 0.24 | 0.06 | 0.22 |

At score >= 0.3: ours LiDAR-only P 0.97 / R 0.43 / F1 0.60, ours Camera+LiDAR 0.98 / 0.42 / 0.59, official Camera+LiDAR 0.93 / 0.92 / 0.93, fused CL+CR 0.60 / 0.97 / 0.74. Our own models are under-confident (scores rarely exceed 0.5), so 0.5 understates them; 0.3 is shown for that reason and was **not** selected on the results.

Recall by range (0-20 / 20-35 / 35-50 m): classical 0.45 / 0.57 / 0.52; official Camera+LiDAR 0.85 / 0.75 / 0.37. The classical pipeline is flat with range (it also sees cluttered near-field structure as objects), the learned model degrades at long range.

## E2. BEV occupancy consistency

`object-cell recall` = share of labelled-object footprint cells covered (1-cell tolerance); `explained` = share of predicted occupied cells lying on a labelled object (1-cell tolerance).

| Source | Object-cell recall | Explained by labelled objects | Occupied cells / frame |
|---|---|---|---|
| Classical LiDAR occupancy (`buildBevOccupancy`) | 0.52 | 0.25 | 991 |
| Classical radar hits (`rasterizeRadarBev`, 5 radars) | 0.31 | 0.28 | 789 |
| Classical LiDAR U radar (plain union) | 0.60 | 0.25 | 1607 |
| Official Camera+LiDAR box footprints (score >= 0.5) | 0.81 | 0.99 | 605 |
| Fused CL+CR box footprints (score >= 0.5) | 0.94 | 0.95 | 839 |
| Ours Camera+LiDAR box footprints | 0.23 | 0.95 | 178 |

**No dense occupancy GT exists in nuScenes-mini**, so "explained" is only a lower bound on precision for the classical maps: buildings, vegetation, poles and walls are real obstacles that are simply not annotated. Do not read 0.25 as "75% false occupancy". The box-footprint rows are near 1.0 by construction (a detected object is a labelled object). This comparison shows what each representation contains, not which is more accurate as an occupancy grid.

## E3. Robustness of the classical pipeline (learned counterpart not measured)

| LiDAR condition | Precision | Recall | F1 | Occupancy object-cell recall | LiDAR path (s / frame) |
|---|---|---|---|---|---|
| full | 0.43 | 0.50 | 0.46 | 0.52 | 0.189 |
| 50 % of points kept | 0.43 | 0.38 | 0.41 | 0.40 | 0.084 |
| 25 % of points kept | 0.44 | 0.30 | 0.35 | 0.29 | 0.100 |
| range limited to 30 m | 0.50 | 0.45 | 0.47 | 0.47 | 0.090 |

Recall falls steadily as LiDAR density falls; precision does not. The same degradation was **not** applied to the learned models (would need re-running inference on degraded input), so no robustness claim is made about them.

## E4. Latency (indicative only, different hardware)

Classical LiDAR path 0.189 s / frame (local CPU, MATLAB with the real toolbox functions; the earlier shim run took 0.519 s); radar rasterisation 0.057 s / frame. Learned official LiDAR-only 0.30 s and Camera+LiDAR 0.52 s per iteration (Tesla T4, batch 1, including data loading, mean of two 50-iteration log windows). CRN latency was not measured in this project. Not a like-for-like comparison.

## Findings

1. On real nuScenes data, the project's unmodified Level-1 classical LiDAR pipeline reaches F1 0.46 (P 0.43, R 0.50). It is workable for vehicles (recall 0.63) and poor for pedestrians (0.18), and roughly half its detections are clusters of walls, vegetation and parked-car groups (its `maxExtent` 8 m / `maxHeight` 2.5 m / pedestrian-size rules were set for a clean simulated street).
2. The pretrained learned pipelines score far higher (F1 0.76-0.90), **but they were trained on trainval, which contains these scenes**, so this is an integration reference, not a fair contest.
3. The fair-ish comparison is against **our own** uncontaminated models: at the pre-declared 0.5 threshold they are worse than classical (F1 0.26-0.32); at 0.3 they are better (F1 ~0.60). That flips with the threshold, so treat "ours vs classical" as inconclusive on 81 samples.
4. Classical maps see everything that reflects LiDAR but know nothing about objects; learned lists know objects but have no static structure or free space. That gap is the motivation for Stage F (learned occupancy).
5. The classical pipeline degrades gracefully but steadily with LiDAR sparsity (recall 0.50 -> 0.38 -> 0.30). Robustness of the learned models was not measured.

## Files

`prepare_inputs.py` (extracts the 81 VAL frames from the local `archive.zip`), `matlab/run_classical_nuscenes.m`, `matlab/shims/`, `evaluate_stage_e.py`, `classical_out/` (toolbox run: occupancy grids, clusters, timings per condition), `classical_out_shim/` (first run with shims), `results/stage_e_metrics.json`, `results/stage_e_compare_sample{5,31,61}.png`, `results/stage_e_detection_bars.png`.
