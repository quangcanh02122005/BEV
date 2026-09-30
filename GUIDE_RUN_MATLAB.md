# Hướng dẫn chạy trên MATLAB

Đã kiểm thử thật ngày 25/09/2026 với MATLAB R2026a (`D:\app\matlab2026a\bin\matlab.exe`) và các toolbox thật (Lidar / Computer Vision / Sensor Fusion…). Kết quả bộ kiểm tra: **MATLAB CHECK PASS (12/12)**.

Quy ước: `WS = D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radat_real_ws`. Mã Level 1 nằm ở `D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radar_ws` (không sửa; các demo tự `addpath` tới đó).
Hệ toạ độ ego: x tiến, y trái, z lên, yaw ngược chiều kim đồng hồ; lưới BEV 200×200, ô 0.5 m, hàng = +x, cột = +y, ±50 m.

## 1. Chạy nhanh: kiểm tra tất cả

Trong MATLAB (hoặc dòng lệnh):
```matlab
cd('D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radat_real_ws'); addpath('matlab');
check_local_matlab
```
```powershell
matlab -batch "cd('D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radat_real_ws'); addpath('matlab'); check_local_matlab"
```
Kiểm tra: dữ liệu `data/` (404 sample, 404 LiDAR, 404 ảnh CAM_FRONT); các file `.mat` (81 frame val, số hộp của official/our/CRN/fused, occupancy `[81 200 200]`); chạy 6 script loader; chạy 2 demo. Mong đợi 12 dòng `PASS`. Mất ~1 phút.

## 2. Chạy pipeline CLASSICAL trên frame bất kỳ của `data/` (thay vì xem ảnh đã lưu)

```matlab
demo_classical_sample('val', 5)       % frame val số 5 (0–80)
demo_classical_sample('train', 100)   % bất kỳ frame train (0–322)
```
Làm gì: đọc trực tiếp `data/v1.0-mini/*.json` và file `.pcd.bin` LiDAR của frame, rồi chạy **đúng mã Level 1**: `processLidarScan` → `buildBevOccupancy` → `extractLidarDetections` (dùng toolbox thật; chỉ khi thiếu toolbox mới tự nạp shim ở `stage_e/matlab/shims`). Vẽ điểm LiDAR, ô occupied (vàng), vật thể classical (cam), GT (xanh lá); với frame val còn hộp Camera+LiDAR official (đỏ) và bản tự train (tím) đọc từ `exports/L3E`.
Đã chạy: `val[5]` → 34 688 điểm, 882 ô occupied, 66 cụm, 28 vật thể (24 xe, 4 người), 2.5 s. Ảnh: `reports\local_runs\matlab_classical_val5.png`.
Ý nghĩa: classical không có ngữ nghĩa ngoài heuristics hình học → nhiều hộp thừa/đứt (xem Stage E trong báo cáo).

Chạy cả 81 frame val + đánh giá: `stage_e\matlab\run_classical_nuscenes.m` (`run_classical_nuscenes('Shims','auto','OutDir',...)`) sau khi `python stage_e\prepare_inputs.py`; đánh giá bằng `python stage_e\evaluate_stage_e.py`.

## 3. Tính lại bản đồ rủi ro (risk) cho một frame

```matlab
demo_risk_frame(51, 'learned_detflow')
demo_risk_frame(51, 'classical')
```
`system` ∈ `oracle | classical | learned_occ | learned_detflow | learned_zeroflow | learned_flownet`. Nạp `stage_h\risk_inputs.mat` (tạo bởi `python stage_h\prepare_risk_inputs.py`), gọi `computeRiskFields` của Level 1, vẽ cạnh oracle (GT occupancy + GT flow), phủ hộp GT (cyan). Đã chạy: frame 51 – max risk 0.431 vs oracle 0.479, ~90 ms cho 2 lần tính. Điểm risk **chưa hiệu chuẩn**, không phải xác suất va chạm.
Chạy toàn bộ Stage H: `stage_h\matlab\run_risk_bridge.m` → `risk_out.mat`, rồi `python stage_h\evaluate_risk.py`.

## 4. Nạp kết quả model (predictions) vào MATLAB

Các script (mỗi cái lưu PNG cạnh script):
| Script | Nội dung |
|---|---|
| `exports\L3E\load_l3e.m` | official LiDAR-only + Camera+LiDAR + GT |
| `exports\L3E\load_camera_radar.m` | CRN Camera+Radar |
| `exports\L3E\check_own.m` | 2 model tự train (LiDAR, Camera+LiDAR) |
| `fusion\unified\load_unified.m` | Unified Object List (late fusion); `source_code` 1=CL, 2=CR, 3=cả hai |
| `models\occupancy\matlab\load_occupancy.m` | occupancy: GT, classical, learned |
| `models\temporal_flow\matlab\load_flow.m` | flow học, flow từ detection, GT |

Tự nạp:
```matlab
S = load('exports/L3E/l3e_predictions.mat');   % S.samples, S.gt, S.pred_lidar_only, S.pred_lidar_camera
P = S.pred_lidar_camera;                        % struct-of-arrays: sample_idx cls x y z w l h yaw vx vy score
k = P.sample_idx == 41 & P.score >= 0.3;        % sample_idx trong .mat là 1-based (val[40] -> 41)
[P.x(k) P.y(k)]
```
Lưu ý: `sample_idx` trong `.mat` bắt đầu từ 1; trong CSV bắt đầu từ 0; trong các script Python `--idx` bắt đầu từ 0.

## 5. MATLAB không chạy được gì
Các checkpoint `.pth` là PyTorch; MATLAB **không** chạy trực tiếp BEVFusion/CRN. Dùng MATLAB để nạp output đã xuất (mục 4) và pipeline classical/risk. Muốn chạy lại trọng số: dùng Python (`GUIDE_RUN_PYTHON_LOCAL.md`) hoặc Kaggle.
Nếu muốn gọi Python từ MATLAB: `pyenv('Version','D:\...\.venv_local\Scripts\python.exe')` rồi `pyrun` / `py.` — chưa kiểm thử trong dự án.

## 6. Sự cố hay gặp
| Triệu chứng | Nguyên nhân / cách xử lý |
|---|---|
| `Using identifier 'grid' as both a variable and a command` | đừng đặt biến tên `grid`, `box`, `cat` khi có lệnh `grid on` (đã tránh trong demo) |
| `Undefined function 'pointCloud'` | thiếu toolbox → demo tự dùng shim; hoặc cài Lidar/Computer Vision Toolbox |
| `Attempt to add "here" to a static workspace` | chạy script loader từ trong function; dùng `run(...)` ở workspace gốc như `check_local_matlab` |
| Ảnh không hiện | các demo dùng `Visible off` và lưu PNG vào `reports\local_runs` |

## 7. Gọi MATLAB từ Python (MATLAB Engine) – đã kiểm thử
Web dùng gói `matlabengine` (`pip install matlabengine` trong `.venv_local`; bản 26.1.12 chạy được với Python 3.11 và MATLAB R2026a Update 5, khởi động ~15 s):
```python
import matlab.engine
eng = matlab.engine.start_matlab()
eng.addpath(r'D:\Vinuni\endgame\cam_lidar_radar\cam_lidar_radat_real_ws\matlab', nargout=0)
```
Hai hàm dành cho việc này: `risk_from_arrays(occ, prob, fx, fy, egoV, yawRate)` và `classical_from_points(pts)`. Đối chiếu: risk live sai khác ≤ 1,5e-8 so với `stage_h\risk_out.mat`; classical live cho 882 ô và 28 vật thể như `demo_classical_sample('val',5)`. Xem `web\README.md`.
