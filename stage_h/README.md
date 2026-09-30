# Stage H — MATLAB risk bridge on real nuScenes-mini frames

The Level-1 Phase-10 function `computeRiskFields` (unmodified, base MATLAB) is fed, frame by frame, with perception outputs from this project's real-data pipeline. **There is no ground-truth risk on nuScenes-mini**, so this stage measures how much perception quality changes the risk score, not whether the risk score is correct. Risk is the project's uncalibrated score in [0, 1], not a collision probability.

## Setup (physical parameters set before running, nothing tuned)

* 81 mini-VAL frames, grid +/-50 m at 0.5 m; ego rectangle 4.084 x 1.730 m (Renault Zoe) instead of the simulation vehicle; grid coordinates shifted by 1.3 m in x so the ego **body centre** is the origin the risk code assumes (the nuScenes ego frame origin is the rear axle). Weights, features, horizon 3 s, tolerances: Level-1 values.
* Ego motion (velocity, yaw rate in ego axes) from the ego poses of the neighbouring keyframes; median ego speed 4.8 m/s. Flow = ground velocity in ego axes (Level-1 definition), NaN treated as zero by the function.
* Six systems, same function: **oracle** (GT occupancy with confidence 1, GT flow); **classical** (`buildBevOccupancy`, no flow); **learned_occ** (`our_occupancy.pth`, no flow); **learned_detflow** (learned static cells U detection footprints of the fused Camera+LiDAR x Camera+Radar list, flow = detection velocities); **learned_zeroflow** (the same occupied cells with flow removed: the project's "static world" baseline); **learned_flownet** (Stage-G network occupancy and its learned flow). Non-oracle occupied cells use the project's uncalibrated confidence 0.9.
* Evaluation follows the project's Phase-10 protocol: object score = highest risk of the system's cells within 0.75 m of the object footprint (0 if none), against the oracle's score for the same object; DCA truth from pure kinematics (independent of the risk formula); high risk = score >= 0.35. 3993 GT objects, 1058 moving, only **24 have oracle risk >= 0.35**.

## Results

Sanity check of the formula on real data: the oracle score correlates with closest-approach proximity (Spearman with -DCA) at **0.85**.

Object-level agreement with the oracle. Most objects have risk near zero (far or parked), which makes plain MAE uninformative, so the table also gives the 1210 objects whose oracle score is >= 0.1 (the project's `oracleRangeMin`; this subset was added after seeing the near-zero dominance, and is declared as such). 95% intervals are bootstrap over the 81 frames.

| System | MAE, all 3993 objects | MAE, oracle >= 0.1 (n = 1210) | Spearman vs oracle, oracle >= 0.1 | Spearman vs -DCA, all |
|---|---|---|---|---|
| classical | 0.018 | 0.029 [0.026, 0.032] | 0.86 [0.83, 0.90] | 0.84 |
| learned occupancy, no flow | 0.014 | 0.023 [0.021, 0.025] | 0.92 [0.90, 0.94] | 0.85 |
| **learned + detection flow** | **0.012** | **0.020 [0.018, 0.022]** | **0.94 [0.93, 0.96]** | 0.87 |
| learned, flow removed | 0.013 | 0.021 [0.019, 0.023] | 0.93 [0.92, 0.95] | 0.85 |
| learned + Stage-G learned flow | 0.022 | 0.044 [0.040, 0.048] | 0.74 [0.69, 0.79] | 0.83 |

Paired bootstrap of MAE differences (oracle >= 0.1): detection flow vs no flow -0.0009 [-0.0013, -0.0005]; detection flow vs classical -0.0088 [-0.0111, -0.0066]; detection flow vs learned flow net -0.0243 [-0.0285, -0.0200].

High-risk objects (24, oracle >= 0.35), precision / recall: classical 0.91 / 0.79; learned occupancy 0.90 / 0.75; learned + detection flow 0.91 / 0.79; flow removed 0.95 / 0.79; **learned flow net 0.20 / 0.83**. These come from 24 objects, so differences of one or two objects are noise.

Flow benefit (the project's test: moving objects, MAE with flow vs flow = 0, required >= 25% relative reduction): 0.0118 vs 0.0147, a **19.9% reduction, below the project's 25% threshold**; on the 336 non-trivial moving objects 0.0247 vs 0.0278 (11%).

Minimum time-to-collision per frame: the oracle predicts a collision in only 14 of 81 frames; where both are finite, MAE is 0.17 s (learned, with or without detection flow), 0.25 s (classical), 1.05 s (learned flow net). Per-frame maximum risk correlates weakly with the oracle for every system (Pearson 0.38-0.46), so frame-level risk ranking is not reliable with this perception.

Runtime: `computeRiskFields` 3-9 ms per frame in MATLAB on the local CPU.

## What this does and does not show

1. With this data, better perception gives a risk map closer to the oracle: learned occupancy beats the classical grid and detection flow adds a small but consistent further gain (about 4% MAE on the non-trivial set, 11% on its moving objects). The gain from flow is small because the ego moves at about 5 m/s, so closing speed and predicted overlap are dominated by ego motion, not by the other object's velocity.
2. **The Stage-G learned flow makes risk worse** (rank correlation 0.74, 80% false high-risk objects). This confirms the Stage-G recommendation: use detection-derived flow, not the learned flow.
3. Caveats that limit every number: the oracle is the same formula on GT inputs whose static part is LiDAR-derived pseudo-GT, and the learned occupancy was trained on that same label definition, which favours it over the classical grid; the detection flow comes from checkpoints trained on trainval (contaminated); confidence is a constant 0.9; 81 frames, 2 scenes; the risk score itself is uncalibrated and unvalidated against real events. These are sensitivity results, not a validation of risk.

## Files

`prepare_risk_inputs.py` -> `risk_inputs.mat`; `matlab/run_risk_bridge.m` -> `risk_out.mat` (risk and TTC maps, six systems); `evaluate_risk.py` -> `results/risk_metrics.json`, `results/stage_h_risk_frame*.png` (oracle vs classical vs learned + detection flow vs learned flow net, on the three frames with the highest oracle risk), `results/stage_h_risk_scatter.png`.
