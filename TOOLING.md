# TOOLING — cách chạy Kaggle từ workspace này

Huấn luyện và suy luận chạy trên **Kaggle (Tesla T4)**; phần đánh giá, xuất, vẽ hình và MATLAB chạy cục bộ. Token Kaggle nằm ở `D:\kaggle_mcp\.env` và **không** được sao chép vào workspace (workspace có thể đưa lên git).

## Lệnh cơ bản (bash, từ thư mục workspace)

```text
tools/kgl.sh kernels status canhquang/<kernel>
tools/kgl.sh kernels push -p notebooks/<kernel-dir>
tools/poll_kernel.sh canhquang/<kernel> [giây]                 # chờ đến khi xong
tools/kgl.sh kernels output canhquang/<kernel> -p <thư mục> [--file-pattern REGEX]   # chỉ tải file cần (env đóng gói rất nặng)
tools/kgl.sh quota                                             # hạn mức GPU tuần (30 h, làm mới 26/09/2026 00:00 UTC)
```

## Các notebook (private, tài khoản `canhquang`)

| Kernel | Mục đích | GPU | Phụ thuộc (`kernel_sources`) |
|---|---|---|---|
| `bev-real-l3c` | BEVFusion pretrained (LiDAR, Camera+LiDAR), dựng env từ đầu | có | - |
| `bev-real-freeze` | dựng lại env, kiểm tra tái lập, đóng gói `env_bevfusion.tar.gz` (3.3 GB) | có | - |
| `bev-real-viz` | 81 ảnh trực quan (6 camera, box 3D, LiDAR BEV) | không | `bev-real-l3c` |
| `bev-real-train-a` / `-b` | Stage A (LiDAR-only) / Stage B (Camera+LiDAR) | có | `freeze`, `l3c` (B thêm `train-a`) |
| `bev-real-radar-audit` | audit dữ liệu radar | không | - |
| `bev-real-cr-env` | env CRN riêng (Python 3.8, torch 1.9.1, CUDA 11.1 chỉ để biên dịch), đóng gói | có | - |
| `bev-real-cr-infer` | CRN pretrained: tạo info/radar cho mini, đánh giá, xuất | có | `cr-env`, `l3c` |
| `bev-real-fusion-eval` | đánh giá devkit cho single-branch và các biến thể fusion | không | `freeze`, `l3c`, `train-b`, `cr-infer` |
| `bev-real-occ` | Stage F: dựng dataset occupancy, train 3 biến thể | có | `freeze` |
| `bev-real-flow` | Stage G: dựng nhãn flow, train 4 biến thể | có | `freeze` |

Mỗi thư mục `notebooks/<kernel>/` có `build_*.py` (sinh `.ipynb` và `kernel-metadata.json`), file script được nhúng vào notebook, và output nhỏ cần cho tái lập trong `out/`.

## Bài học vận hành (đã gặp thật)

- Notebook Kaggle chạy **từ đầu** mỗi lần đẩy phiên bản mới; env (conda) mất sau mỗi lần chạy. Vì vậy env được đóng gói thành `env_*.tar.gz` và mount qua `kernel_sources` (khôi phục mất khoảng 4-10 phút thay vì 15-30 phút dựng lại).
- `MPLBACKEND` của Jupyter làm matplotlib trong env conda hỏng: notebook luôn `export MPLBACKEND=Agg`.
- Checkpoint pretrained BEVFusion chỉ nạp đúng khi có `spconv-cu120` (trọng số dùng layout spconv); luôn kiểm tra log không có `size mismatch`.
- Evaluator của mmdet3d lấy `version` từ metainfo của dataset: đặt `test_dataloader.dataset.metainfo.version=v1.0-mini`.
- Tạo file lớn bằng công cụ ghi file, **không** dùng heredoc lồng nhau trong shell (đã làm vỡ lệnh nhiều lần).
- Sinh notebook từ script Python (`build_*.py`) rồi `kernels push`; đọc log qua `kernels output` sau khi xong (không xem được log khi đang chạy).

## Giả định đường dẫn tuyệt đối còn lại

- `stage_e\prepare_inputs.py` đọc `..\docs\archive.zip` (đường dẫn tuyệt đối trong file).
- `stage_e\matlab\run_classical_nuscenes.m` trỏ tới `..\cam_lidar_radar_ws` của Level 1 (chỉ đọc, `addpath` thêm `src` và `config`).
- Các script còn lại dùng đường dẫn tương đối theo cấu trúc workspace.
