# NEXT_ACTIONS

## Còn dở sau Stage H

1. **Runtime thống nhất:** đo lại latency / FPS / bộ nhớ GPU của BEVFusion (LiDAR, Camera+LiDAR), CRN, hai U-Net trên cùng một phiên T4 (hiện có số rời rạc ở phần cứng khác nhau; chưa có latency CRN).
2. **Robustness của model learned:** áp cùng mức suy giảm LiDAR (giữ 50% / 25% điểm, giới hạn tầm 30 m) cho suy luận learned để so với classical (mới đo cho classical).
3. **Own Camera+Radar training** (`our_camera_radar.pth`): optional; cần driver Lightning riêng cho CRN R18.
4. Rà lại báo cáo cuối cùng với người dùng trước khi trình bày (`reports/LEVEL3_FINAL_REPORT.md`, mục 7 là danh sách điều không được tuyên bố).

## Sau khi Level 3 PASS (không làm trước)

- Flow học lại với chuỗi **sweep** (cách nhau 0.05-0.1 s) thay vì keyframe cách 0.5 s, thêm tương quan chuyển động; chỉ giữ nếu vượt zero-flow và vượt flow suy ra từ detection.
- Own Camera+Radar training (`our_camera_radar.pth`, CRN R18, ResNet-18 ImageNet, cần driver Lightning riêng).
- Train lâu hơn / tối ưu Stage A/B (sau khi có bằng chứng cần thiết).
