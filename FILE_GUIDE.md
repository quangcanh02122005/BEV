# Chức năng của từng file trong workspace

`D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radat_real_ws` – Level 3 (Advanced Research Lite) trên nuScenes-mini thật. Bảng theo thứ tự thư mục. Chạy thử: `GUIDE_RUN_PYTHON_LOCAL.md`, `GUIDE_RUN_MATLAB.md`.

Nguyên tắc chung: nuScenes-mini = 404 keyframe (323 train / 81 val; val = scene-0103, scene-0916). Checkpoint official huấn luyện trên trainval nên **nhiễm** trên mini-val; đây không phải benchmark đầy đủ. Định dạng chung: `sample_idx,class,x,y,z,w,l,h,yaw,vx,vy,score` trong hệ ego (x tiến, y trái, z lên).

## Gốc
| File | Chức năng |
|---|---|
| `README.md` | Tổng quan, cấu trúc, cách dùng nhanh |
| `PROJECT_STATE.md` | Bảng trạng thái các Stage A–H (đã xong/kết quả/giới hạn) |
| `NEXT_ACTIONS.md` | Việc còn lại: đo runtime hệ hợp nhất, độ bền với LiDAR thưa, tự train Camera+Radar (tuỳ chọn), báo cáo review |
| `TOOLING.md` | Lệnh Kaggle, bảng notebook, bài học rút ra |
| `FILE_GUIDE.md` / `GUIDE_RUN_PYTHON_LOCAL.md` / `GUIDE_RUN_MATLAB.md` | Ba hướng dẫn này |
| `.gitignore` | Bỏ `*.pth, *.npz, *.tar.gz, **/out/, .env, .venv_local/` (checkpoint được theo dõi bằng sha256) |
| `.venv_local/` | Python venv local (torch CPU) để chạy `tools/local` |
| `data/` | Dữ liệu nuScenes-mini bạn tải: `samples/`, `sweeps/`, `maps/`, `v1.0-mini/*.json` |

## `models/` – trọng số
| Đường dẫn | Chức năng |
|---|---|
| `MODEL_ZOO.md` | Danh mục mọi model: nguồn, dữ liệu huấn luyện, chỉ số, cảnh báo |
| `CHECKPOINTS.sha256` | sha256 của 21 file `.pth` (kiểm bằng `inspect_checkpoints.py`) |
| `lidar_only/our_lidar.pth` | BEVFusion LiDAR-only tự train từ đầu trên mini-train |
| `camera_lidar/our_bevfusion.pth` | BEVFusion Camera+LiDAR tự train (khởi tạo từ LiDAR ở bước A) |
| `camera_radar/` | CRN R18 pretrained (chạy trên Kaggle), `radar_audit.json` (kiểm tra radar), README, file env |
| `occupancy/our_occupancy.pth` (+`variants/`) | U-Net occupancy 3 lớp (free/static/object); biến thể chỉ lidar, chỉ radar; `matlab/` chứa `.mat` + `load_occupancy.m` |
| `temporal_flow/our_flow.pth` (+`variants/`) | U-Net flow T4 (có warp ego); biến thể T1, T2, T4_nowarp; `matlab/` |
| `our_bevfusion_mini/` | Artifact gốc gộp (README 4 hàng so sánh, `epoch_ckpts/`, `predictions/`) |

## `exports/L3E/` – kết quả detector ở dạng dùng chung
| File | Chức năng |
|---|---|
| `pred_lidar_only_ego.csv`, `pred_lidar_camera_ego.csv` | Official BEVFusion LiDAR-only / Camera+LiDAR (81 val) |
| `pred_camera_radar_ego.csv` | CRN Camera+Radar |
| `pred_our_lidar_ego.csv`, `pred_our_bevfusion_ego.csv` | Hai model tự train |
| `gt_ego.csv`, `samples.csv` | GT và danh sách frame (sample_idx ↔ token) |
| `l3e_predictions.mat`, `camera_radar_predictions.mat`, `own_models_predictions.mat` | Bản `.mat` cho MATLAB (struct-of-arrays, sample_idx 1-based) |
| `export_l3e.py`, `export_camera_radar.py`, `export_own_models.py` | Sinh CSV/.mat từ output thô trên Kaggle |
| `load_l3e.m`, `load_camera_radar.m`, `check_own.m` | Loader/plot MATLAB |
| `tables/` | Bản sao bảng nuScenes json dùng khi xuất |

## `fusion/` – Stage D, hợp nhất muộn 3 cảm biến
| File | Chức năng |
|---|---|
| `late_fusion.py` | Ghép Camera+LiDAR với Camera+Radar → Unified Object List (`source_code` 1/2/3) |
| `export_unified.py` | Xuất unified CSV + `.mat` cho pretrained và bản tự train |
| `op_points.py` | Chọn điểm vận hành (ngưỡng) → `unified/operating_points.json` |
| `unified/` | `unified_pretrained.*`, `unified_ours.*`, `load_unified.m` |

## `stage_e/` – Classical vs Learned
| File | Chức năng |
|---|---|
| `prepare_inputs.py` | Tách 81 frame val từ `docs/archive.zip` vào `stage_e/data` |
| `matlab/run_classical_nuscenes.m` | Chạy pipeline Level 1 (classical) trên 81 frame, tuỳ chọn `'Shims'`, `'OutDir'` |
| `matlab/shims/` | Bản thay thế toolbox khi máy thiếu; `probe_*.m` là script dò toolbox |
| `evaluate_stage_e.py` | Tính chỉ số classical vs learned |
| `classical_out/` | Kết quả chạy toolbox thật; `classical_out_shim/` là chạy bằng shim (khớp nhau) |
| `results/` | `stage_e_metrics*.json` và hình |

## `stage_f/` – Learned occupancy
| File | Chức năng |
|---|---|
| `build_occ_dataset.py` | Dựng input 7 kênh (LiDAR + radar) và nhãn free/static/object; nhãn static = pseudo-GT LiDAR ≥4/11 sweep, z∈(0.3, 3.0) m; object = footprint GT thật |
| `train_occ.py`, `build_occ_nb.py` | Huấn luyện trên Kaggle và dựng notebook |
| `evaluate_stage_f.py` | IoU, so classical |
| `out/outputs/` | Checkpoint, `occ_val_predictions.npz`, metrics |
| `results/` | `stage_f_metrics.json` (mIoU val 0.549), `occupancy_predictions.mat`, hình |

## `stage_g/` – Temporal / flow
| File | Chức năng |
|---|---|
| `build_flow_dataset.py` | Chuỗi T frame, ma trận warp ego (A, b), nhãn flow = vận tốc mặt đất |
| `train_flow.py`, `build_flow_nb.py` | Huấn luyện (regress flow/10), dựng notebook |
| `evaluate_stage_g.py` | EPE, so zero-flow / không warp |
| `out/outputs/`, `results/` | Output val, `stage_g_reference.json`, `flow_predictions.mat`, hình |

## `stage_h/` – Đánh giá cuối bằng risk
| File | Chức năng |
|---|---|
| `prepare_risk_inputs.py` | Gom occupancy/flow của từng hệ + ego velocity + GT vật thể → `risk_inputs.mat` |
| `matlab/run_risk_bridge.m` | Gọi `computeRiskFields` (Level 1) cho mọi hệ → `risk_out.mat` |
| `evaluate_risk.py` | So sánh risk giữa các hệ với oracle → `results/risk_metrics.json`, hình |

## `matlab/` (mới) – chạy thật trên máy
| File | Chức năng |
|---|---|
| `demo_classical_sample.m` | Classical Level 1 trên frame bất kỳ đọc từ `data/` |
| `demo_risk_frame.m` | Tính/vẽ risk cho 1 frame val theo hệ chọn |
| `check_local_matlab.m` | Bộ kiểm tra tổng (12 mục) |
| `risk_from_arrays.m`, `classical_from_points.m` | Hai hàm web gọi qua MATLAB Engine (risk và classical trên mảng/điểm đưa vào) |

## `web/` – giao diện web (mới)
| File | Chức năng |
|---|---|
| `app.py` | Giao diện Streamlit: chọn frame, các mục Camera / LiDAR+Radar / Occupancy / Flow / Detection / Tracking / Risk / Classical / Ablation / Mô phỏng Level 1 / Chỉ số |
| `panels.py` | Tính và vẽ từng chức năng (tái dùng `tools/local`) |
| `tables.py` | Dựng bảng ablation từ các file kết quả có sẵn |
| `tracking.py` | Tracker tự viết + đo MOTA-lite so GT |
| `matlab_bridge.py` | Gọi MATLAB Engine: risk live, classical live |
| `README.md` | Cách chạy, nguồn kết quả (live hay tính sẵn), giới hạn |

## `tools/`
| File | Chức năng |
|---|---|
| `kgl.sh`, `poll_kernel.sh` | Gọi Kaggle CLI (token từ `D:\kaggle_mcp\.env`) và chờ kernel chạy xong |
| `local/nuscenes_local.py` | Đọc nuScenes-mini không cần devkit; dựng input BEV, nhãn, flow |
| `local/models.py` | Kiến trúc UNet + nạp checkpoint + warp |
| `local/run_occupancy.py`, `run_flow.py` | Suy luận local trên frame bất kỳ; `--all` kiểm parity với Kaggle |
| `local/view_detections.py` | Vẽ hộp dự đoán trên ảnh camera thật + LiDAR BEV |
| `local/check_data.py`, `inspect_checkpoints.py` | Kiểm tra dữ liệu / trọng số |

## `notebooks/` – Kaggle
`build_*.py` sinh `.ipynb` + `kernel-metadata.json` cho các kernel: `bev-real-l3c` (pretrained), `freeze`, `viz`, `train-a/b` (LiDAR, Camera+LiDAR), `radar-audit`, `cr-env`, `cr-infer` (CRN), `fusion-eval`, `occ`, `flow`. Thư mục `out/` nhỏ giữ log/kết quả.

## `reports/`
| File | Chức năng |
|---|---|
| `LEVEL3_FINAL_REPORT.md` | Báo cáo cuối, số liệu và giới hạn |
| `FINAL_CHECKLIST_STATUS.md` | Checklist hoàn thành |
| `make_overview.py` → `level3_overview.png` | Ảnh tổng quan |
| `local_runs/` | PNG do các script local/MATLAB tạo (có thể xoá) |

## `frozen/L3C_L3E_v1/`
Baseline đã đóng băng của L3C–L3E (chỉ đọc, có `SHA256SUMS`). Không sửa.
