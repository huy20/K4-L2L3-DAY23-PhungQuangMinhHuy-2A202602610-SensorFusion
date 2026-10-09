# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phung Quang Minh Huy
- MSSV: 2A202602610
- Email: minhhuy.phung03@gmail.com
- Link repo (fork): https://github.com/huy20/K4-L3-Day23-PhungQuangMinhHuy-2A202602610
- Commit hash nộp (`git rev-parse HEAD`): 15e70c7cca0ec372a9c529c7cb9d155e8056d735

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
  `compare`; `frames = [0, 198]`; `seed = 0`;
  `segment = training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
  `precision = 0.9701`, `recall = 0.7004`, `tp = 519`, `fp = 16`, `fn = 222`
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `rmse = 0.1503 m`, `matches = 502`, `sum_sq_err = 11.3369`, `ghost_track_frames = 0`,
  `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
  `rmse = 0.1358 m`, `matches = 502`, `sum_sq_err = 9.2642`, `ghost_track_frames = 0`,
  `missed_gt_frames = 239`, `mean_confirmed_tracks = 2.5226`
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Hai mode có **cùng** `matches = 502`, `ghost_track_frames = 0`, `missed_gt_frames = 239`
  và `mean_confirmed_tracks ≈ 2.523`, tức camera **không** tạo/xoá/đổi danh tính track —
  đúng thiết kế track-then-fuse. Khác biệt duy nhất là vị trí: `sum_sq_err` giảm từ
  11.3369 (lidar) xuống 9.2642 (fused), nên RMSE giảm từ 0.1503 m xuống 0.1358 m
  (fused tốt hơn ~0.0144 m, thoả điều kiện `rmse_fused − rmse_lidar ≤ 0.05`).
  Chất lượng track: `precision_track = matches/(matches+ghosts) = 502/502 = 1.0`,
  `coverage = matches/det_tp = 502/519 = 0.967`. `missed_gt_frames = 239` chủ yếu do
  detector bỏ sót (`fn = 222`), không phải ghost. Vì vậy phải đọc RMSE **cùng** số ghép,
  ghost và miss: RMSE thấp nhưng matches ít hoặc ghost nhiều vẫn mất điểm.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?**
   Lidar: `z` là vị trí 3D `(x, y, z)` trong hệ vehicle. `kalman.py` dùng `H` tuyến tính
   3×6 `[I₃ | 0]` (platform `Sensor` cho lidar là identity), `R = diag(0.1², 0.1², 0.1²)`
   (3×3). Camera: `z` là toạ độ pixel 2D `(u, v)`; `camera_measurement_prediction` chiếu
   pinhole `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (phi tuyến), nên `H` là
   **Jacobian** 2×6 (đạo hàm phép chiếu × phép quay, platform cung cấp), `R = diag(5², 5²)`
   (2×2) tính từ `sigma_cam_i/j`. Khác số chiều nên ngưỡng gate χ² cũng khác `dim_meas`.

2. **Vì sao cần gating Mahalanobis trước khi gán?**
   Mahalanobis `d² = γᵀ S⁻¹ γ` với `γ = z − h(x)`, `S = H P Hᵀ + R`
   (`association.mahalanobis_distance` + `innovation`/`innovation_covariance`) chuẩn hoá
   residual theo **bất định** của cả dự đoán lẫn đo. Khoảng cách Euclid bỏ qua `P`: khi
   `P` lớn (track mới/chưa chắc) dự đoán kém chính xác nên residual lớn vẫn hợp lệ; ngược
   lại khi `P` nhỏ thì residual nhỏ mà lệch đã đáng ngờ. `chi2_gate` loại cặp có `d²`
   vượt `chi2.ppf(gating_threshold, dim_meas)`, giảm ghép sai trước khi gán greedy.
   Cặp **ngoài FOV** bị loại trước cả bước này (`association_cost_matrix` kiểm tra
   `meas.sensor.in_fov(track.x)` trước khi tính Mahalanobis) để không chiếu điểm vô lệ.

3. **Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.**
   Là **track-then-fuse**: một tracker duy nhất, mỗi frame `predict` một lần rồi cập nhật
   EKF lần lượt lidar (AssocL) → camera (AssocC). Thấy trong `run_lab.py`: vòng `for track
   in manager.track_list: KF.predict(track)` rồi `associate_and_update(..., lidar_sensor)`
   rồi `associate_and_update(..., camera_sensor)`. Trên `grade_run.log`, số `confirmed`
   và danh tính track do lidar quyết định; hai mode có `matches/ghosts/misses` giống nhau
   (502/0/239) nhưng `sum_sq_err` khác → camera chỉ tinh chỉnh trạng thái chứ không đổi
   vòng đời track.

4. **Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?**
   Sai extrinsic/intrinsics làm `h(x)` (và Jacobian `H`) lệch, nên `γ = z − h(x)` có
   **thành phần hệ thống** (không còn trung bình ~0) ngay cả với vật được track tốt; các
   cặp camera đúng có thể bị `chi2_gate` loại (hoặc ghép nhầm). `S = H P Hᵀ + R` cũng
   sai theo. Hệ quả: `rmse_fused` xấu đi (có thể vượt `rmse_lidar`), `sum_sq_err` tăng,
   innovation camera lớn bất thường. Đây là lý do camera chỉ được tinh chỉnh state, không
   tham gia quyết định tồn tại track.

5. **Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?**
   Mỗi lượt sensor kết thúc bằng `manager.manage_tracks(unassigned_tracks,
   unassigned_meas, sensor)` **kể cả khi `meas_list` rỗng**. Với lidar, frame 0 detection
   vẫn phải trừ score các track nằm trong FOV (miss) và xoá track cạn điểm; với camera,
   lượt rỗng phải không làm gì. `sensor` là thứ phân biệt hai luật đó (`manage_tracks`
   trả về sớm nếu `sensor.name != "lidar"`). Lidar quyết định `init/score/delete` vì nó cho
   vị trí 3D đáng tin để xác nhận/xoá; camera 2D (không depth) chỉ cập nhật EKF state,
   không cộng/trừ score, không sinh, không xoá track.

6. **Điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xoá track.**
   `update_track_score`: hit cộng `1/window` (cap 1), miss trong FOV trừ `1/window`;
   xác nhận khi `score > confirmed_threshold` (0.8), track đã confirmed thì **không** hạ
   trạng thái vì một miss. `should_delete_track`: xoá khi `P[0,0]` hoặc `P[1,1] > max_P`
   (3² = 9), **hoặc** confirmed có `score < delete_threshold` (0.6), **hoặc** chưa
   confirmed có `score ≤ 0`. Camera không gọi các hàm này.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Kilo (trợ lý lập trình AI, model ds/deepseek-v4-flash)
- Dùng cho phần nào (hàm, câu hỏi, debug): hỗ trợ viết code Part E–H
  (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) và soạn nháp
  báo cáo này.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy
  `pytest student/tests -q` (128 passed, không failed/xfailed); chạy
  `fusion-run-lab --fusion compare --seed 0` trên segment mặc định; đối chiếu công thức
  `F/Q`, `γ = z − h(x)`, `S = HPHᵀ + R`, `K = PHᵀS⁻¹`, gating χ² và mô hình pinhole với
  code và test gốc.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (không sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
