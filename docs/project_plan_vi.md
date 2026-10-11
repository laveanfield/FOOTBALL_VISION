# Football Vision: Các giai đoạn & quy trình thực hiện đồ án Computer Vision

> **Môn học:** Computer Vision  
> **Hướng đề tài:** *Football Tactical Vision Analyzer* — từ video bóng đá: phát hiện & theo dõi cầu thủ, ước lượng hình học sân, chiếu vị trí lên bản đồ sân 2D nhìn từ trên xuống để phân tích chiến thuật.  
> **Phiên bản:** 1.0 · cập nhật 04/10/2026  
> **Quy ước thời gian:** "Tuần N" là tuần *tương đối* kể từ ngày nhóm bắt đầu. Khi có hạn nộp thật, co/giãn theo mục 7.

## Cách đọc tài liệu

| Bạn cần… | Xem mục |
|---|---|
| Hiểu đề đòi gì và đề tài đáp ứng thế nào | 1 |
| Chốt phạm vi, mức độ (Level), nguồn dữ liệu | 2 |
| Kiến trúc pipeline & định dạng dữ liệu giữa các module | 3 |
| **Các giai đoạn thực hiện (trọng tâm)** | 4 |
| Quy trình làm việc nhóm, git, theo dõi thí nghiệm | 5 |
| Phân công, lịch, mốc kiểm soát | 6, 7 |
| Kế hoạch đánh giá & tối ưu | 8 |
| Báo cáo PDF, cấu trúc repo | 9 |
| Rủi ro, checklist nộp bài, câu hỏi bảo vệ | 10, 11, 12 |
| Toán nền tảng (suy diễn từ đầu), template sân, code mẫu, tài liệu | Phụ lục A–D |

---

## Tóm tắt 1 phút

**Bài toán.** Input: video bóng đá. Output: (i) video overlay (bbox + ID + vạch sân); (ii) bản đồ sân 2D với vị trí cầu thủ theo thời gian; (iii) phân tích chiến thuật (heatmap, team shape, quãng đường/tốc độ).

**Quy trình 6 bước**

1. **Chốt phạm vi**: đề tài, Level mục tiêu (Level 2 là sàn, Level 3 là đích), nguồn dữ liệu, phân vai.
2. **Dựng walking skeleton**: pipeline thô chạy end-to-end ngay tuần đầu để mọi module cùng cắm vào một khung.
3. **Phát triển theo Level**: xong Level 1 → 2 → 3; mỗi mức có mốc kiểm soát và một bản demo chạy được.
4. **Vòng lặp từng module**: giả thuyết → cài đặt → *nhìn bằng mắt* → đo → ghi chép.
5. **Đánh giá & so sánh**: số liệu (mAP, IDF1/MOTA/HOTA, reproj error, FPS) + ablation + phân tích lỗi.
6. **Viết báo cáo song song**: xong giai đoạn nào viết 1–2 trang giai đoạn đó; đóng băng code ≥ 1 tuần trước hạn.

**Điểm mạnh khi bảo vệ.** Phần CV truyền thống (Canny/Hough → homography → căn chỉnh frame) do nhóm tự hiện thực và giải thích được từ nguyên lý; phần deep learning (YOLO, ByteTrack) dùng pretrained nhưng có đủ *vai trò – giả định – hạn chế* và có số đo.

---

## 1. Phân tích đề bài

### 1.1 Đề yêu cầu gì (tóm tắt)

- **Hình thức:** nhóm 3–4 SV; đề *mở* (tự chọn dữ liệu, phạm vi, mức độ phức tạp) nhưng bắt buộc vận dụng tổng hợp kiến thức CV; bài làm *xuyên suốt môn học* nên phải lập kế hoạch và làm từ sớm.
- **Bốn thành phần cốt lõi:**
  - **3.1** Thu thập & tiền xử lý dữ liệu (resize/crop, đổi không gian màu, giảm nhiễu/cân bằng sáng).
  - **3.2** Phân tích hình học & cấu trúc cảnh *ít nhất một* trong: biên/đường thẳng/góc; ước lượng biến đổi affine/projective; ghép/căn chỉnh ảnh; ước lượng mặt phẳng.
  - **3.3** Phân tích nội dung/đối tượng *ít nhất một* kỹ thuật (detect, track, phân vùng cơ bản, salient regions). Được dùng model pretrained nhưng phải giải thích vai trò, giả định, hạn chế.
  - **3.4** Trực quan hóa: bbox/mask/keypoints/đường/mặt phẳng; so sánh trực quan input vs output; với video thì trình bày theo trình tự thời gian.
- **Mở rộng (khuyến khích):** so sánh nhiều phương pháp; đánh giá định tính/định lượng; tối ưu tốc độ/ổn định; demo tương tác đơn giản.
- **Sản phẩm:** báo cáo PDF (bài toán & dữ liệu; pipeline; phân tích & thảo luận kết quả; chỉ code chính) + repo public chứa code đầy đủ + mô tả phân công; mỗi thành viên có đóng góp kỹ thuật rõ ràng.

### 1.2 Ma trận truy vết yêu cầu (yêu cầu → module → bằng chứng)

| Mục đề | Yêu cầu | Đáp ứng bằng | Giai đoạn | Bằng chứng đưa vào báo cáo |
|---|---|---|---|---|
| 3.1 | Thu thập dữ liệu | Clip bóng đá (nguồn công khai / tự quay) + ground truth tối thiểu | 1 | Bảng mô tả dữ liệu, nguồn, giấy phép |
| 3.1 | Resize/crop, đổi màu, giảm nhiễu | OpenCV: resize 1280×720, BGR→RGB/Gray/HSV/Lab, CLAHE, Gaussian | 1 | Ảnh before/after |
| 3.2 | Biên, đường thẳng, góc | Canny + Hough + giao điểm | 2 | Lưới ảnh: gốc → biên → đường → keypoints |
| 3.2 | Biến đổi projective | Homography ảnh→sân (DLT + RANSAC + LM) | 3 | Công thức, reproj error, overlay template |
| 3.2 | Căn chỉnh nhiều ảnh | Căn chỉnh frame↔frame (feature matching + homography) | 3 | Ảnh warp, đồ thị drift |
| 3.2 | Ước lượng mặt phẳng | Mặt phẳng sân (mask màu + homography) | 1, 3 | Mask sân, ảnh warp top-down |
| 3.3 | Phát hiện đối tượng | YOLO pretrained (± fine-tune) | 4 | mAP, hình TP/FP/FN |
| 3.3 | Theo dõi đối tượng | ByteTrack + SORT tự cài | 5 | IDF1/MOTA/HOTA, quỹ đạo |
| 3.3 | Phân vùng cơ bản (tuỳ chọn) | Phân vùng sân/đội bằng màu (HSV, KMeans) | 1, 6 | Mask, nhãn đội |
| 3.4 | Hiển thị bbox/đường/keypoints/mặt phẳng | Overlay bbox+ID, vạch sân, top-down | 8 | Video + khung hình |
| 3.4 | So sánh input vs output; theo thời gian | Video side-by-side; dải khung hình theo thời gian | 8 | Hình dải thời gian |
| 4 | So sánh phương pháp | Ablation A1–A8 (mục 8.2) | 9 | Bảng so sánh |
| 4 | Đánh giá định lượng | mAP, IDF1, MOTA, HOTA, reproj error, FPS | 9 | Bảng số liệu |
| 4 | Tối ưu tốc độ/ổn định | FP16, imgsz, skip-frame, làm mượt H | 9 | Đồ thị speed–accuracy, jitter |
| 4 | Demo tương tác | Streamlit/Gradio | 8 | Ảnh chụp/link demo |
| 5, 6 | Báo cáo + repo + phân công | Mục 9 và mục 6 của tài liệu này | 10 | PDF, repo, bảng đóng góp |

### 1.3 Vì sao hướng bóng đá phù hợp

Một video bóng đá chứa đủ bốn lớp bài toán trong cùng một cảnh: **hình học sân** (đường thẳng, mặt phẳng, phép chiếu phối cảnh), **phát hiện** (người, bóng), **theo dõi** (ID theo thời gian) và **trực quan hóa** (top-down, heatmap). Nhờ vậy phần "CV truyền thống" có chỗ đứng tự nhiên, không bị thu hẹp thành "gọi YOLO rồi vẽ bbox".

### 1.4 Ba nguyên tắc thiết kế (để an toàn khi bảo vệ)

1. **Không chỉ gọi model.** Phần hình học (Canny/Hough, homography, căn chỉnh frame) và ít nhất một thành phần tracking (SORT tối giản) do nhóm tự hiện thực và giải thích được từ nguyên lý.
2. **Mỗi module có bộ ba Vai trò – Giả định – Hạn chế.** Đề yêu cầu đúng điều này khi dùng model pretrained (mục 3.3 của đề).
3. **Mỗi kết luận đi kèm số liệu và ví dụ lỗi.** Định lượng cho "tốt đến đâu", định tính cho "hỏng khi nào, vì sao".

---

## 2. Phạm vi, mức độ & dữ liệu

### 2.1 Bốn mức độ (Level)

| Level | Nội dung | Kết quả nhìn thấy | Vai trò trong kế hoạch |
|---|---|---|---|
| 1 | YOLO phát hiện người + Canny/Hough vạch sân + overlay | Video có bbox và đường sân | Mốc **M1**: sàn an toàn |
| 2 | + ByteTrack + homography + chiếu top-down + heatmap | Bản đồ sân 2D có vị trí cầu thủ, heatmap | Mốc **M2**: **MVP bắt buộc** |
| 3 | + phân loại đội + quỹ đạo + team shape + quãng đường/tốc độ | Phân tích chiến thuật theo đội | Mốc **M3**: **mục tiêu chính** |
| 4 | + tracking bóng + possession + pass detection + passing network | Mini football analytics | Chỉ chọn 1 hạng mục (possession) nếu còn thời gian |

**Khuyến nghị: chiều sâu trước chiều rộng.** Dành thời gian cho đánh giá, so sánh phương pháp và phân tích lỗi thay vì thêm tính năng Level 4. Lý do: (i) phần "kết quả mong đợi" của đề nhấn vào năng lực thiết kế – kết nối kiến thức – phân tích đánh giá, không đếm số tính năng; (ii) Level 4 phụ thuộc vào bóng (thành phần khó và kém ổn định nhất).

### 2.2 Những trường hợp cần xem xét để xử lý

| # | Một số trường hợp | Vấn đề thực tế | Cách xử lý trong kế hoạch này |
|---|---|---|---|
| 1 | "YOLO pretrained detect player / ball / referee / goalkeeper" | YOLO pretrain COCO chỉ có lớp `person` và `sports ball`; không phân biệt cầu thủ / thủ môn / trọng tài | Phase 4 phát hiện `person`; Phase 6 gán vai trò bằng màu áo + vị trí; fine-tune lớp vai trò là tuỳ chọn (thành một ablation) |
| 2 | "Hough → keypoints → homography" tự động | Trên video broadcast: sọc cỏ, bóng đổ, che khuất, zoom làm Hough kém ổn định; vòng tròn giữa sân thành ellipse | Phase 3 dùng *bán tự động + lan truyền* làm baseline; Hough dùng để *kiểm tra/hiệu chỉnh drift* và làm phương pháp tự động để so sánh |
| 3 | "Distance covered / speed" | Homography chỉ đúng cho điểm nằm trên mặt sân; broadcast chỉ thấy một phần sân nên ID/quỹ đạo đứt; đạo hàm khuếch đại nhiễu | Báo cáo là *ước lượng trên đoạn nhìn thấy*; làm mượt trước khi đạo hàm; có sanity check (Phase 7, Phụ lục A7) |
| 4 | "Possession / pass detection / passing network" | Bóng nhỏ, hay mất, bay trên không (vi phạm giả định mặt phẳng) | Để ở Level 4, tuỳ chọn; chỉ làm possession nếu ball tracking đạt ngưỡng |
| 5 | "mAP / IDF1 / MOTA / homography error" | Cần ground truth | Lập kế hoạch GT ngay Phase 1 (hoặc dùng dataset công khai) |
| 6 | "Video 2–5 phút" | Đủ cho demo, nhưng broadcast có cut/replay/zoom nên homography đứt đoạn | Phase 1 thêm tách shot; chỉ phân tích các shot "wide" |

### 2.3 Lựa chọn nguồn dữ liệu

| Phương án | Mô tả | Độ khó homography | Bản quyền / giấy phép | Khuyến nghị |
|---|---|---|---|---|
| **A. Broadcast** (camera chính, chuyển động) | Video truyền hình / clip công khai | Khó nhất: $H_t$ đổi từng frame (pan/tilt/zoom), có cut/replay | Rủi ro: thường có bản quyền | Dùng làm *clip phụ* cho phần căn chỉnh frame và phân tích lỗi |
| **B. Camera cố định / tactical view** (từ dataset công khai) | Góc rộng cố định, thấy gần toàn sân | Dễ: một $H$ cho cả clip | Tuỳ dataset (đọc điều khoản) | Hợp làm *clip chính* để chắc chắn đạt Level 2–3 |
| **C. Tự quay** (nơi cao cố định như khán đài/tầng cao; drone nếu được phép) | Nhóm chủ động | Dễ–vừa (camera cố định) | An toàn nhất (xin phép người trong khung) | Rất tốt nếu có điều kiện; nên đo/ghi kích thước sân thật |

**Chiến lược hai clip:** clip chính (B hoặc C — $H$ cố định hoặc ít zoom) bảo đảm Level 2–3; clip phụ (A) để trình diễn căn chỉnh frame và phân tích lỗi.

**Nguồn tham khảo:** SoccerNet (video, calibration, tracking), SportsMOT (MOT cho thể thao) — *đọc trang chính thức về điều khoản, đăng ký, NDA trước khi dùng*. Mọi nguồn phải được ghi trong báo cáo. Không đưa video thô vào repo public; chỉ đưa script tải + ảnh/GIF minh hoạ ngắn.

---

## 3. Kiến trúc pipeline & hợp đồng dữ liệu

### 3.1 Sơ đồ pipeline (nhãn P = số giai đoạn ở mục 4)

```text
                        VIDEO ĐẦU VÀO
                              │
                 [P1] Tiền xử lý · tách shot · mask sân
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
   [P2] HÌNH HỌC SÂN                     [P4] PHÁT HIỆN ĐỐI TƯỢNG
 Canny · Hough · giao điểm                  YOLO (người, bóng)
        │                                           │
        ▼                                           ▼
 [P3] HOMOGRAPHY                            [P5] TRACKING
 DLT+RANSAC · căn chỉnh frame↔frame      ByteTrack · SORT tự cài
 làm mượt · kiểm tra drift                          │
        │                               [P6] ĐỘI · VAI TRÒ · BÓNG
        │                                           │
        └──────────────────┬────────────────────────┘
                           ▼
   [P7] CHIẾU TOP-DOWN: điểm chân → toạ độ sân (mét)
                           ▼
               [P7] PHÂN TÍCH CHIẾN THUẬT
   heatmap · quỹ đạo · team shape · quãng đường/tốc độ · possession
                           ▼
   [P8] TRỰC QUAN HOÁ & DEMO ───► [P9] ĐÁNH GIÁ · SO SÁNH · TỐI ƯU
```

### 3.2 Hợp đồng dữ liệu giữa các module

Thống nhất định dạng ngay từ đầu để các thành viên làm song song; module hạ nguồn dùng dữ liệu giả/GT khi module thượng nguồn chưa xong.

| Artefact | Nội dung / schema | Sinh bởi | Dùng bởi |
|---|---|---|---|
| `meta.json` | fps, width, height, n_frames, danh sách shot (start, end) | P1 | tất cả |
| `pitch_mask/` | mask nhị phân mặt sân (H×W) mỗi frame/shot | P1 | P2, P4 |
| `pitch_lines.csv` | frame_id, x1, y1, x2, y2, orient, score | P2 | P3 |
| `homography.npy` | mảng (T,3,3) float64, ảnh→sân (mét), chuẩn hoá H[2,2]=1 | P3 | P7 |
| `homography_quality.csv` | frame_id, valid, n_inliers, reproj_err_m, overlay_score | P3 | P7, P9 |
| `detections.csv` | frame_id, x1, y1, x2, y2, conf, cls | P4 | P5 |
| `tracks.csv` | frame_id, track_id, x1, y1, x2, y2, conf, team, role | P5, P6 | P7 |
| `pitch_xy.csv` | frame_id, track_id, X, Y (mét), team, role, valid | P7 | P8, P9 |
| `analytics/` | heatmap, team shape theo thời gian, bảng tốc độ/quãng đường | P7 | P8, P9 |
| `outputs/`, `reports/figures/` | video, hình, bảng cuối | P8, P9 | báo cáo |

### 3.3 Quy ước

- **Toạ độ ảnh:** pixel, gốc ở góc trên-trái, $x$ sang phải, $y$ xuống dưới; frame đã resize 1280×720.
- **Toạ độ sân:** mét, gốc ở góc trên-trái của bản đồ sân; $X$ chạy dọc chiều dài (0–105), $Y$ chạy dọc chiều rộng (0–68).
- `frame_id` bắt đầu từ 0; thời gian $t = \text{frame\_id}/\text{fps}$.
- $H$ luôn là **ảnh → sân**; nghịch đảo $H^{-1}$ là sân → ảnh (dùng để vẽ template lên video).

### 3.4 Nguyên tắc kỹ thuật

- Mỗi giai đoạn = một stage đọc artefact vào, ghi artefact ra → chạy lại riêng từng stage, không phải chạy lại YOLO khi chỉ sửa homography.
- Mọi tham số nằm trong file config (YAML); seed cố định; log thời gian chạy từng stage.
- *Fail-soft:* stage lỗi ở một frame thì đánh dấu `valid=0` và đi tiếp, không làm sập cả pipeline.

---

## 4. Các giai đoạn thực hiện

### Tổng quan các giai đoạn

Cột "Lead" ứng với phân vai 4 người ở mục 6 (A: Dữ liệu & Hình học · B: Homography · C: Detection–Tracking–Đội · D: Phân tích–Trực quan–Tích hợp).

| Phase | Tên | Tuần | Lead | Đầu ra chính | Mốc |
|---|---|---|---|---|---|
| 0 | Khởi động & chốt phạm vi | 1 | cả nhóm | repo skeleton, ma trận yêu cầu, walking skeleton | M0 |
| 1 | Dữ liệu & tiền xử lý | 1–2 | A | clip, shot list, mask sân, GT tối thiểu | M0 |
| 2 | Hình học sân (biên, đường, keypoints) | 2–4 | A | vạch sân + keypoints, so sánh phương pháp | M1 |
| 3 | Homography & căn chỉnh theo thời gian | 3–5 | B | $H_t$ + chỉ số chất lượng | M2 |
| 4 | Phát hiện đối tượng | 3–5 | C | `detections.csv`, mAP | M1 |
| 5 | Tracking | 4–6 | C | `tracks.csv`, IDF1/MOTA/HOTA | M2 |
| 6 | Đội, vai trò, bóng | 5–7 | C (+D) | nhãn đội, bóng (tuỳ chọn) | M3 |
| 7 | Top-down & phân tích chiến thuật | 6–8 | D | `pitch_xy.csv`, heatmap, shape, tốc độ | M3 |
| 8 | Trực quan hoá & demo | 7–9 | D | video, hình, demo | M4 |
| 9 | Đánh giá, so sánh, tối ưu | 8–9 | cả nhóm | bảng số liệu, ablation | M4 |
| 10 | Báo cáo, repo, bảo vệ | 9–10 | cả nhóm | PDF, repo, slide | M5 |

Mỗi giai đoạn dưới đây theo cùng khuôn: **Mục tiêu → (Ý tưởng cốt lõi) → Quy trình → Đầu ra → Tiêu chí hoàn thành (DoD) → Rủi ro**. Phần suy diễn toán đầy đủ nằm ở Phụ lục A.

---

### Phase 0: Khởi động & chốt phạm vi (Tuần 1)

**Mục tiêu.** Cả nhóm hiểu đề giống nhau, chốt đề tài/Level/dữ liệu, và có *walking skeleton* chạy end-to-end (dù thô) trước khi đào sâu từng module.

**Quy trình**

1. Đọc đề cùng nhau; điền ma trận yêu cầu (mục 1.2); ghi rõ nhóm chọn những hướng nào trong "ít nhất một trong…" của 3.2 và 3.3.
2. Chốt: đề tài (Football Tactical Vision Analyzer), Level mục tiêu (Level 2 là sàn, Level 3 là đích), nguồn dữ liệu (mục 2.3).
3. Phân vai chính/phụ cho từng module (mục 6).
4. Tạo repo **public** ngay: README skeleton, cấu trúc thư mục (mục 9.3), `.gitignore` (`data/`, `weights/`), license.
5. Dựng môi trường tái lập: Python ≥ 3.10, `requirements.txt` hoặc `environment.yml` (OpenCV, NumPy, SciPy, pandas, matplotlib, scikit-learn, PyTorch, Ultralytics, thư viện tracker, Streamlit). Xác định tài nguyên GPU (máy cá nhân, Colab, Kaggle).
6. Viết **walking skeleton**: `scripts/run_pipeline.py` đọc 10 s video → YOLO → ghi video có bbox. Mục đích: mọi người chạy được cùng một lệnh và thống nhất I/O.
7. Lập *risk register* v1 (mục 10) và lịch họp cố định.

**Đầu ra.** Repo + README skeleton; ma trận yêu cầu; bảng phân công; walking skeleton; risk register.

**DoD.** Mọi thành viên clone và chạy được skeleton trên máy mình; có board công việc cho Phase 1–3.

---

### Phase 1: Thu thập & tiền xử lý dữ liệu (Tuần 1) · *đề 3.1*

**Mục tiêu.** Có tập clip sạch, được mô tả đầy đủ, kèm mask sân và ground truth tối thiểu để đánh giá về sau.

**Quy trình**

1. **Chọn clip** (mục 2.3): clip dev 30–60 s; clip test 1–2 phút (không tuning trên test); clip "khó" (zoom, đông người, ánh sáng xấu) cho phân tích lỗi. Tiêu chí: thấy rõ vạch sân, ≥ 720p, 25–30 fps, ít cut/replay, ánh sáng đều.
2. **Ghi metadata** (nguồn, giấy phép, fps, độ phân giải, thời lượng) vào `data/README.md`; không commit video lớn (lưu link/script tải).
3. **Tách shot** (với broadcast): khoảng cách histogram HSV giữa các frame liên tiếp, đỉnh vượt ngưỡng = cut → danh sách shot; loại close-up/replay; chỉ ước lượng homography trong shot "wide".
4. **Tiền xử lý** (lưu ảnh before/after cho báo cáo): decode bằng `cv2.VideoCapture`; resize 1280×720 (`INTER_AREA` khi thu nhỏ); BGR→RGB (OpenCV đọc BGR); grayscale cho Canny/Hough; HSV cho mask sân; CLAHE trên kênh L (Lab) khi ánh sáng không đều; Gaussian/bilateral để giảm nhiễu trước khi lấy đạo hàm.
5. **Mask mặt sân** bằng màu: ngưỡng HSV quanh hue xanh lá (tự thích nghi bằng histogram kênh H hoặc KMeans trên (H,S)) → morphological open/close → thành phần liên thông lớn nhất. Mask dùng để lọc nhiễu cho Hough và lọc detection ngoài sân.
6. **Tạo ground truth tối thiểu** (CVAT hoặc Label Studio):
   - *Detection:* 100–200 frame (player/GK/referee/ball) từ clip dev + test.
   - *Homography:* 30–50 frame, click ≥ 6 landmark **không dùng để ước lượng $H$**.
   - *Tracking:* đoạn 10–20 s theo định dạng MOT (frame, id, x, y, w, h); tăng tốc bằng cách pre-annotate từ tracker rồi sửa tay.
   - Chia dev/test theo **clip** (không theo frame) để tránh rò rỉ dữ liệu.

**Đầu ra.** `data/` có cấu trúc; shot list; mask sân; GT v0; hình before/after.

**DoD.** Có bảng thống kê dữ liệu (số clip, thời lượng, số frame, số bbox GT); pipeline đọc + tiền xử lý chạy ổn định và không phải nút cổ chai.

**Rủi ro.** Bản quyền (→ mục 2.3). Thiếu thời gian annotate (→ chia đều theo thành viên; chấp nhận GT nhỏ nhưng nêu rõ giới hạn thống kê).

---

### Phase 2: Phân tích hình học sân (Tuần 2–4) · *đề 3.2: biên, đường thẳng, góc*

**Mục tiêu.** Trích xuất đường sân và điểm đặc trưng (giao điểm) từ ảnh — nền cho homography.

**Ý tưởng cốt lõi.** Vạch sân là đường thẳng trắng trên nền cỏ. Canny cho biên; Hough gom các điểm biên thẳng hàng thành đường thẳng bằng *bỏ phiếu* trong không gian $(\rho,\theta)$; giao điểm hai đường tính bằng tích có hướng trong toạ độ đồng nhất (Phụ lục A1–A2).

**Quy trình**

1. **Biên.** Canny trên ảnh xám đã mask sân; thử ≥ 2 cấu hình ($\sigma$, tỉ lệ ngưỡng cao:thấp ≈ 2:1–3:1); ghi lại khác biệt định tính.
2. **Đặc trưng vạch trắng (so sánh).** (a) Canny; (b) ngưỡng sáng + morphological *top-hat* để chỉ giữ cấu trúc sáng và hẹp (vạch), bỏ sọc cỏ. So sánh số đường đúng/sai giữa (a) và (b).
3. **Hough.** `HoughLinesP` (hoặc Hough chuẩn + gom nhóm); chỉnh `threshold`, `minLineLength`, `maxLineGap`. Hậu xử lý: hợp nhất đoạn gần thẳng hàng theo $(\rho,\theta)$; phân nhóm hai họ đường song song ngoài đời (dọc/ngang sân) theo hướng hoặc theo vanishing point (RANSAC).
4. **Vòng tròn giữa sân.** Dưới phối cảnh nó là ellipse nên `HoughCircles` không phù hợp; dùng `fitEllipse` trên thành phần biên, hoặc bỏ qua và chỉ dùng đường thẳng (nêu rõ trong báo cáo).
5. **Giao điểm → keypoints ứng viên.** $\mathbf{p}=\mathbf{l}_1\times\mathbf{l}_2$; lọc điểm nằm trong khung hình và gần mask sân.
6. **Ghép với template sân** (Phụ lục B): hai chế độ, cấp dữ liệu cho Phase 3:
   - *Bán tự động:* click landmark ở keyframe đầu mỗi shot.
   - *Tự động (hypothesize-and-verify):* chọn 2 đường "ngang" + 2 đường "dọc" ↔ cặp tương ứng trên template; 4 giao điểm cho 4 cặp điểm → tính $H$; chấm điểm bằng độ khớp của các đường template còn lại với bản đồ biên; giữ giả thuyết điểm cao nhất.
7. **Trực quan hoá.** Lưới 4 ô: ảnh gốc | biên | đường Hough | keypoints  (dùng thẳng cho báo cáo).

**Đầu ra.** `pitch_lines.csv`; hình pipeline hình học; danh sách lỗi (sọc cỏ, bóng đổ, che khuất).

**DoD.** Trên clip dev, đường dọc/ngang chính được phát hiện ở phần lớn frame (báo cáo tỉ lệ trên subset đã gán nhãn); có ≥ 1 so sánh phương pháp (Canny vs top-hat).

**Rủi ro.** Sọc cỏ/bóng đổ tạo biên giả → dùng mask + top-hat + lọc theo độ dài/hướng. Zoom/nhoè chuyển động → dùng chế độ bán tự động và lan truyền ở Phase 3.

---

### Phase 3: Homography & căn chỉnh theo thời gian (Tuần 3–5) · *đề 3.2: projective, căn chỉnh ảnh, mặt phẳng*

**Mục tiêu.** Với mỗi frame hợp lệ, ước lượng $H_t$: ảnh → sân (mét), kèm chỉ số chất lượng.

**Ý tưởng cốt lõi.** Mặt sân là một mặt phẳng; ảnh của mặt phẳng qua camera pinhole là phép biến đổi projective (homography, 8 bậc tự do): $H=K[\mathbf{r}_1\ \mathbf{r}_2\ \mathbf{t}]$. Mỗi cặp điểm tương ứng cho 2 phương trình ⇒ cần ≥ 4 cặp (không 3 điểm thẳng hàng). Khi camera chỉ xoay (pan/tilt/zoom), homography giữa hai frame bất kỳ *chính xác cho mọi điểm cảnh* (Phụ lục A3–A4).

**Quy trình**

1. **Template sân** 105×68 m (Phụ lục B) + hàm vẽ template lên ảnh.
2. **Camera cố định:** ước lượng $H$ một lần từ ≥ 6–8 landmark (RANSAC + tinh chỉnh LM), áp dụng cho toàn clip.
3. **Camera chuyển động (broadcast):**
   1. Keyframe đầu mỗi shot: $H_k$ từ landmark (bán tự động hoặc tự động từ Phase 2).
   2. Mỗi frame: tìm đặc trưng nền (ORB/SIFT/AKAZE) **ngoài** bbox người → khớp (ratio test) → `findHomography(RANSAC)` ra $G_t$ (frame $t$ → frame $t-1$) → $H_t=H_{t-1}G_t$. Biến thể ít drift hơn: căn chỉnh thẳng về keyframe, $H_t=H_kG_{t\to k}$.
   3. **Chống drift.** Sai số nhân dồn qua chuỗi; đo bằng *line-overlay score* (chiếu template về ảnh bằng $H_t^{-1}$ rồi so với bản đồ khoảng cách của biên Canny). Khi điểm thấp hơn ngưỡng, hoặc mỗi N frame, *re-anchor* bằng $H$ từ đường Hough (Phase 2). Đây là chỗ Hough/Canny "đóng vòng" với homography.
   4. (Tuỳ chọn) tinh chỉnh bằng căn chỉnh trực tiếp: `findTransformECC` với `MOTION_HOMOGRAPHY`.
4. **Làm mượt theo thời gian.** Không trung bình trực tiếp các phần tử ma trận; làm mượt vị trí ảnh của 4 điểm tham chiếu (ví dụ 4 góc một hình chữ nhật trên sân) rồi dựng lại $H$ từ 4 điểm (`getPerspectiveTransform`).
5. **Kiểm định & đánh dấu hợp lệ.** (i) reproj error trên landmark held-out; (ii) line-overlay score; (iii) sanity: định thức/số điều kiện của $H$, ảnh của 4 góc khung hình nằm trong vùng hợp lý. Frame vi phạm → `valid=0` (không đưa vào analytics).
6. **Trực quan hoá.** (a) Vẽ template lên video, đường sân phải "dính" đúng vạch; (b) `warpPerspective` frame sang top-down để kiểm tra bằng mắt; (c) đồ thị sai số theo thời gian.

**Đầu ra.** `homography.npy` (T,3,3), `homography_quality.csv`; video overlay; đồ thị reproj error.

**DoD.** Báo cáo reproj error theo **mét** (mean/median/p95) trên landmark held-out; tỉ lệ frame hợp lệ trong shot "wide"; overlay không "bơi" giữa các frame. Mục tiêu tham khảo (điều chỉnh theo thực tế): median dưới ~1 m trên sân, ≥ ~90% frame hợp lệ.

**Rủi ro.** Drift (→ re-anchor); zoom nhanh (→ SIFT/AKAZE bền hơn ORB, hoặc bỏ đoạn đó); ít vạch nhìn thấy (< 2 đường mỗi họ không đủ ràng buộc → giữ $H$ từ lan truyền); $H$ suy biến (→ cờ `valid=0`).

---

### Phase 4: Phát hiện đối tượng (Tuần 3–5, song song Phase 2–3) · *đề 3.3*

**Mục tiêu.** Detection ổn định cho người (và bóng), có đánh giá định lượng và hiểu rõ giới hạn.

**Quy trình**

1. **Baseline.** YOLO pretrained trên COCO (lớp `person` = 0, `sports ball` = 32). Ghi rõ họ model, phiên bản, trọng số. Chạy ở `imgsz` lớn (≥ 960–1280) vì cầu thủ trong cảnh rộng rất nhỏ.
2. **Hậu xử lý.** Ngưỡng confidence, NMS; lọc kích thước bbox; lọc theo mask sân (điểm chân phải nằm trong sân ± biên) để loại khán giả/HLV/ballboy.
3. **Vai trò (player / GK / referee).** COCO không có các lớp này ⇒ hai hướng: (a) fine-tune trên dataset bóng đá có nhãn lớp, hoặc trên GT tự gán; (b) gán vai trò bằng heuristic ở Phase 6. Làm (b) trước, (a) nếu còn thời gian → so sánh.
4. **Bóng.** Nhỏ, mờ, hay bị che ⇒ thử SAHI/tiling hoặc crop động; nội suy khi mất ngắn; loại outlier theo tốc độ tối đa; ghi lại tỉ lệ frame có bóng.
5. **Fine-tune (tuỳ chọn).** Chia train/val/test theo clip; augmentation (scale, mosaic, color jitter nhẹ); so sánh pretrained vs fine-tuned.
6. **Đánh giá.** Precision/recall, mAP@0.5, mAP@0.5:0.95 theo lớp (trên GT Phase 1); độ trễ và FPS cho các cỡ model (nano/small/medium…) → đồ thị speed–accuracy.

**Đầu ra.** `detections.csv`; bảng mAP; hình TP/FP/FN; đồ thị speed–accuracy.

**DoD.** Detection chạy toàn clip; phân tích 5–10 ca lỗi có chú thích; điền bộ ba Vai trò–Giả định–Hạn chế (mục 9.2).

**Rủi ro.** Cầu thủ ở xa quá nhỏ → tăng `imgsz`, tiling. Domain gap COCO ↔ bóng đá → fine-tune nhẹ. Bóng gần như không bắt được → hạ Level 4 xuống tuỳ chọn.

---

### Phase 5: Tracking (Tuần 4–6) · *đề 3.3*

**Mục tiêu.** Gán ID ổn định cho từng cầu thủ theo thời gian và thu được quỹ đạo.

**Ý tưởng cốt lõi.** *Tracking-by-detection* gồm ba bước: (1) dự đoán vị trí track từ trạng thái trước (Kalman, vận tốc đều); (2) ghép dự đoán với detection mới bằng bài toán gán tối ưu (Hungarian) trên chi phí $1-\mathrm{IoU}$; (3) cập nhật trạng thái. ByteTrack thêm một vòng ghép thứ hai dùng cả detection điểm thấp để "cứu" track bị che (Phụ lục A5).

**Quy trình**

1. **Baseline ByteTrack** (bản chính thức hoặc cài sẵn trong thư viện) → `tracks.csv`. Chỉnh ngưỡng điểm cao/thấp, `track_buffer`, `match_thresh` trên **dev**.
2. **Tự hiện thực SORT tối giản** (~100–150 dòng: Kalman + `scipy.optimize.linear_sum_assignment`) để hiểu cơ chế và có phần CV "của nhóm" → so sánh với ByteTrack (IDF1, IDSW).
3. **So sánh phương pháp:** ByteTrack vs SORT-tự-cài vs (tuỳ chọn) tracker dùng appearance (DeepSORT).
4. **Tracking trong toạ độ sân.** Dùng $H_t$ chiếu điểm chân sang mét, chạy Kalman với trạng thái $[X,Y,\dot X,\dot Y]$. *Giả thuyết:* khi camera pan mạnh, chuyển động ảnh của cầu thủ = chuyển động thật + chuyển động do camera nên vi phạm giả định vận tốc đều; trong toạ độ sân, chuyển động camera đã được $H_t$ khử ⇒ ít ID switch hơn. Đây là ablation nổi bật (image-space vs pitch-space).
5. **Hậu xử lý.** Xoá track quá ngắn; nối tracklet (khoảng trống ngắn + cùng đội + liên tục trong toạ độ sân); làm mượt quỹ đạo.
6. **Đánh giá.** MOTA, IDF1, HOTA, IDSW, Frag trên đoạn GT (py-motmetrics hoặc TrackEval).

**Đầu ra.** `tracks.csv`; bảng so sánh tracker; hình quỹ đạo.

**DoD.** ID ổn định trong phần lớn đoạn dev; bảng chỉ số; ≥ 1 phân tích nguyên nhân ID switch (che khuất? màu áo giống? camera?).

**Rủi ro.** Nhiều ID switch khi cầu thủ chen nhau → tracklet stitching, tracking trong toạ độ sân, appearance cue. Track đứt khi ra khỏi khung → chấp nhận và ghi vào giới hạn.

---

### Phase 6: Phân loại đội, vai trò & bóng (Tuần 5–7) · *Level 3–4*

**Mục tiêu.** Gắn nhãn đội/vai trò cho từng track; (tuỳ chọn) tracking bóng và possession.

**Quy trình**

1. **Phân loại đội.** Cắt vùng thân trên (khoảng 15–55% chiều cao bbox, bỏ rìa), loại pixel xanh sân bằng mask → đặc trưng màu (median hoặc histogram trong Lab/HSV) → KMeans (k=2 cho cầu thủ trường; k=3–4 nếu gộp GK/trọng tài) → **bỏ phiếu theo track** (đa số trên nhiều frame) để ổn định nhãn.
2. **Vai trò.** GK = cụm màu riêng + vị trí gần khung thành trong thời gian dài; trọng tài = cụm màu thứ ba/ít người. Báo cáo độ chính xác trên subset GT.
3. **So sánh phương pháp (tuỳ chọn).** Màu + KMeans vs embedding ảnh (mô hình pretrained) + clustering.
4. **Possession (Level 4, tuỳ chọn).** Cầu thủ gần bóng nhất trong toạ độ sân (ngưỡng bán kính + thời lượng tối thiểu $k$ frame để tránh nhấp nháy) → đội đang giữ bóng. *Cảnh báo:* bóng bay vi phạm giả định mặt phẳng nên vị trí chiếu của bóng sai.
5. **Pass detection / passing network (Level 4, tuỳ chọn).** Chỉ làm nếu tracking bóng tốt; chuyền = đổi người sở hữu trong cùng đội; mạng chuyền = đồ thị có hướng có trọng số.

**Đầu ra.** Cột `team`, `role` trong `tracks.csv`; (nếu làm) bảng possession.

**DoD.** Nhãn đội cho đa số track; accuracy/confusion matrix trên subset GT; nếu làm possession thì ghi rõ giới hạn.

**Rủi ro.** Hai đội áo gần màu, áo sọc → dùng không gian Lab + embedding. Trọng tài/GK nhiễu cụm → k lớn hơn + luật vị trí.

---

### Phase 7: Chiếu top-down & phân tích chiến thuật (Tuần 6–8) · *đề 3.4 + mở rộng*

**Mục tiêu.** Chuyển kết quả thị giác thành số liệu chiến thuật trên mặt sân.

**Ý tưởng cốt lõi.** Chỉ điểm nằm *trên mặt sân* mới được homography ánh xạ đúng ⇒ dùng **điểm chân** (giữa cạnh dưới bbox), không dùng tâm bbox. Đạo hàm khuếch đại nhiễu ⇒ phải làm mượt trước khi tính vận tốc/quãng đường (Phụ lục A7).

**Quy trình**

1. **Chiếu.** $\mathbf{P}_{i,t}=\pi\big(H_t\,\tilde{\mathbf{p}}^{\text{foot}}_{i,t}\big)$; loại điểm ngoài sân quá biên cho phép (đánh dấu `valid`).
2. **Làm mượt quỹ đạo.** Savitzky–Golay hoặc Kalman smoother theo từng track (chỉ trong đoạn liên tục).
3. **Heatmap.** Histogram 2D ô 1×1 m + Gaussian ($\sigma$ chọn theo nhiễu vị trí, cỡ vài mét); chuẩn hoá theo **thời gian nhìn thấy** của từng track/đội (vì track biến mất khi ra khỏi khung).
4. **Team shape.** Centroid, độ dài $L$, độ rộng $W$, diện tích convex hull, stretch index (Phụ lục A8) theo thời gian; có tuỳ chọn loại GK.
5. **Quãng đường & tốc độ.** $D_i=\sum_t\lVert\Delta\mathbf{P}_{i,t}\rVert$, $v=\lVert\Delta\mathbf{P}\rVert/\Delta t$ (m/s; × 3.6 = km/h) **sau khi làm mượt**; chia vùng tốc độ (đi bộ / chạy chậm / chạy / nước rút — ngưỡng tham khảo từ tài liệu phân tích thể thao, có biến thể theo nguồn); gắn cờ lỗi khi tốc độ > ~12 m/s.
6. **Giới hạn phải nêu.** Chỉ đo trên phần nhìn thấy; kích thước sân thật có thể khác 105×68 m; sai số $H$ kéo theo sai số vị trí rồi sai số vận tốc.

**Đầu ra.** `pitch_xy.csv`; thư mục `analytics/` (heatmap, shape, bảng tốc độ); hình top-down.

**DoD.** Video top-down khớp video gốc (kiểm tra bằng mắt ở ≥ 3 đoạn); số liệu qua sanity check (phân bố tốc độ hợp lý, không có "dịch chuyển tức thời").

**Rủi ro.** Điểm chân bị che/cắt khỏi khung → bbox lệch → vị trí sai (đánh dấu `valid`). Sai số $H$ lớn ở vùng xa camera → báo cáo sai số theo khoảng cách.

---

### Phase 8: Trực quan hoá & demo (Tuần 7–9) · *đề 3.4 + mở rộng*

**Mục tiêu.** Biến kết quả thành hình/video dễ hiểu, đúng yêu cầu 3.4 (input vs output, theo trình tự thời gian).

**Sản phẩm cần có**

1. **Video side-by-side:** trái = video gốc + bbox/ID/màu đội + vạch sân từ $H_t$; phải = bản đồ top-down với chấm cầu thủ, vệt quỹ đạo (≤ 50 frame gần nhất), bóng.
2. **Dải khung hình theo thời gian** (filmstrip: frame tại $t_0, t_0+\Delta, \dots$ kèm top-down tương ứng) — đáp ứng "trình bày theo trình tự thời gian".
3. **Hình pipeline từng bước:** gốc → tiền xử lý → biên → đường → keypoints → warp top-down.
4. **Hình phân tích:** heatmap (cầu thủ/đội), team shape (hull + centroid), biểu đồ tốc độ, đồ thị lỗi homography theo thời gian.
5. **Thư viện lỗi (failure gallery):** 6–10 ca lỗi điển hình có chú thích nguyên nhân.
6. **Demo tương tác (tuỳ chọn):** Streamlit/Gradio — chọn clip mẫu, bật/tắt lớp (bbox, vạch sân, top-down, quỹ đạo, heatmap), thanh trượt thời gian, hiển thị số đo chính.

**Mẹo kỹ thuật.** Màu cố định theo đội; ghi video bằng `cv2.VideoWriter` (mp4v) rồi chuyển H.264 bằng ffmpeg nếu cần phát trên trình duyệt; giữ nhất quán hệ toạ độ top-down; xuất GIF 5–10 s cho README.

**DoD.** 1–2 video demo; bộ hình đủ cho báo cáo; demo chạy bằng một lệnh.

---

### Phase 9: Đánh giá, so sánh & tối ưu (Tuần 8–9) · *mở rộng (mục 4 của đề)*

**Mục tiêu.** Chứng minh hệ thống tốt đến đâu, vì sao, và hỏng khi nào.

**Quy trình**

1. Đóng băng cấu hình trên dev; chạy đánh giá trên **test** một lần (mục 8.1).
2. Chạy các ablation A1–A8 (mục 8.2) — mỗi ablation: 1 giả thuyết, 1 biến thay đổi, 1 metric, 1 kết luận.
3. Profiling & tối ưu tốc độ (mục 8.4); báo cáo FPS kèm phần cứng cụ thể.
4. Phân tích lỗi có hệ thống: phân loại lỗi (che khuất, màu áo giống, zoom, replay, ánh sáng, bóng đổ), mỗi loại 1–2 ví dụ + nguyên nhân + thử khắc phục.
5. Chốt bảng/hình cuối bằng script (`scripts/eval_*.py`) để tái lập.

**DoD.** Mọi con số trong báo cáo sinh ra từ script + config + commit hash; không còn "số làm tay".

---

### Phase 10: Báo cáo, repo & bảo vệ (Tuần 9–10) · *đề mục 5–6*

**Quy trình**

1. **Viết song song:** nháp từng phần ngay khi giai đoạn xong (mục 5.8); tuần 9 ráp thành bản v1; tuần 10 đọc chéo (người khác đọc phần của mình).
2. **Báo cáo PDF** theo outline mục 9.1; chỉ chèn code chính; có bảng đóng góp từng thành viên.
3. **Repo public:** README, hướng dẫn cài/chạy, cấu trúc (mục 9.3), link dữ liệu/trọng số, GIF demo; gắn tag `v1.0`.
4. **Kiểm thử trên máy sạch:** một thành viên *không viết* pipeline clone repo và chạy theo README.
5. **Chuẩn bị bảo vệ:** slide 10–15 trang; kịch bản demo 3–5 phút + video dự phòng; diễn tập Q&A (mục 12); mỗi người trình bày phần mình làm.

**DoD.** Checklist mục 11 hoàn tất.

---

## 5. Quy trình làm việc nhóm & kỹ thuật

### 5.1 Vòng lặp phát triển cho mỗi module

```text
Giả thuyết ─► Cài đặt ─► Trực quan hoá (debug bằng mắt) ─► Đo (metric) ─► Ghi chép
    ▲                                                                       │
    └──────────────────────────── điều chỉnh ◄──────────────────────────────┘
```

Nguyên tắc: **nhìn trước, đo sau.** Trong CV, một overlay sai lộ ra ngay khi vẽ lên ảnh, trong khi một con số trung bình có thể che mất nó. Mỗi thay đổi đều có một hình và một số đi kèm.

### 5.2 Chiến lược tích hợp

- **Walking skeleton → làm dày dần:** pipeline end-to-end chạy từ tuần 1; mỗi module thay thế dần phần "giả" bằng phần thật.
- **Dữ liệu giả để chạy song song:** module hạ nguồn dùng GT/giả lập (ví dụ $H$ cố định, track từ GT) khi module thượng nguồn chưa xong.
- **Integration run hàng tuần** trên clip dev vào cuối tuần; không để tích hợp dồn về cuối.

### 5.3 Git / GitHub flow

- `main` luôn chạy được; nhánh `feat/<phase>-<chủ đề>`; mọi PR cần ≥ 1 review từ thành viên khác.
- Commit message dạng `[phase3] add RANSAC homography`; mỗi người commit phần của mình — đây là bằng chứng đóng góp mà đề yêu cầu.
- Gắn tag theo mốc: `m1`, `m2`, `m3`, `v1.0`.

### 5.4 Quản lý công việc

- Kanban 4 cột (Backlog / Doing / Review / Done); mỗi task có *owner*, *reviewer*, *DoD*.
- Nhịp tuần: đầu tuần lập kế hoạch 15 phút; giữa tuần check-in bất đồng bộ; cuối tuần demo + integration run 30–45 phút.

### 5.5 Theo dõi thí nghiệm & tái lập

- Mỗi lần chạy dùng một file config (YAML); seed cố định; lưu `metrics.json` kèm commit hash.
- Bảng `experiments.md`: Ngày · Người · Thay đổi · Config · Metric · Nhận xét.
- Cache artefact trung gian (mục 3.2) để chạy lại từng stage.
- (Tuỳ chọn) DVC cho dữ liệu/artefact; MLflow hoặc Weights & Biases cho theo dõi thí nghiệm.

### 5.6 Definition of Done cho một task

Code chạy từ CLI/config · có hình minh hoạ · có số đo (nếu áp dụng) · có docstring/README ngắn · đã được review · không làm hỏng pipeline end-to-end.

### 5.7 Cổng chất lượng ở mỗi mốc

Pipeline chạy được trên bản clone sạch · số liệu cập nhật · hình lưu ở `reports/figures/` · đoạn báo cáo của giai đoạn đã có nháp (1–2 trang).

### 5.8 Viết báo cáo song song

Xong một giai đoạn → người phụ trách viết ngay 1–2 trang gồm: phương pháp, kết quả, và bộ ba Vai trò–Giả định–Hạn chế. Tránh dồn viết vào tuần cuối; đề cũng nói rõ phải thực hiện sớm ngay từ đầu kỳ.

### 5.9 "Học tới đâu làm tới đó"

Đề là bài xuyên suốt môn học, nên ánh xạ giai đoạn vào bài giảng (tuỳ lịch thực tế của môn):

| Nội dung bài giảng (điển hình) | Giai đoạn có thể bắt đầu |
|---|---|
| Xử lý ảnh cơ bản, không gian màu, lọc | Phase 1 |
| Biên & đường thẳng (Canny, Hough) | Phase 2 |
| Biến đổi hình học, homography, đặc trưng & khớp ảnh | Phase 3 |
| Phát hiện đối tượng / deep learning | Phase 4 |
| Chuyển động & theo dõi | Phase 5 |

---

## 6. Phân công nhóm

### 6.1 Phương án 4 thành viên

| Vai | Giai đoạn chính | Tự hiện thực (không chỉ gọi thư viện) | Đánh giá phụ trách | Mục báo cáo |
|---|---|---|---|---|
| **A: Dữ liệu & Hình học** | 1, 2 | tách shot, mask sân, gom/lọc đường Hough, giao điểm | IoU mask, tỉ lệ đường/keypoint đúng | §2, §4.1–4.2 |
| **B: Homography & Camera** | 3 | DLT + chuẩn hoá (đối chiếu `cv2`), lan truyền $H_t$, re-anchor, làm mượt | reproj error (m), overlay score, jitter | §4.3 |
| **C: Detection, Tracking & Đội** | 4, 5, 6 | SORT tối giản, tracklet stitching, phân loại đội | mAP, MOTA/IDF1/HOTA, accuracy đội | §4.4–4.6 |
| **D: Phân tích, Trực quan & Tích hợp** | 7, 8 (+ điều phối 9, 10) | chiếu top-down, heatmap, team shape, tốc độ/quãng đường, renderer, demo | sanity check, FPS hệ thống | §3, §4.7, §5 |

(§ = mục của báo cáo, xem mục 9.1.)

### 6.2 Phương án 3 thành viên

Gộp **A + B** thành "Dữ liệu – Hình học – Homography" (đây là chuỗi phụ thuộc chặt nhau); C và D giữ nguyên. Nếu quá tải, hạ Phase 6 (bóng, possession) xuống tuỳ chọn và rút ngắn demo tương tác.

### 6.3 Nguyên tắc phân công

- **Cross-review:** A ↔ B, C ↔ D; PR của người này do người cặp còn lại duyệt.
- **Backup owner:** mỗi module có một người dự phòng (giảm rủi ro một người kẹt/bận).
- **Công bằng và rõ ràng (đề yêu cầu):** mỗi người có ≥ 1 module "tự hiện thực", ≥ 1 chỉ số đánh giá riêng, ≥ 2 mục báo cáo, và lịch sử commit/PR tương ứng.

### 6.4 Mẫu bảng đóng góp cho báo cáo

| Thành viên | MSSV | Vai trò | Module phụ trách | Đóng góp kỹ thuật chính (commit/PR tiêu biểu) | Mục báo cáo |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

---

## 7. Lịch trình & mốc kiểm soát

### 7.1 Gantt rút gọn (10 tuần tương đối)

```text
Tuần  01 02 03 04 05 06 07 08 09 10
P0    ██ ·· ·· ·· ·· ·· ·· ·· ·· ··
P1    ██ ██ ·· ·· ·· ·· ·· ·· ·· ··
P2    ·· ██ ██ ██ ·· ·· ·· ·· ·· ··
P3    ·· ·· ██ ██ ██ ·· ·· ·· ·· ··
P4    ·· ·· ██ ██ ██ ·· ·· ·· ·· ··
P5    ·· ·· ·· ██ ██ ██ ·· ·· ·· ··
P6    ·· ·· ·· ·· ██ ██ ██ ·· ·· ··
P7    ·· ·· ·· ·· ·· ██ ██ ██ ·· ··
P8    ·· ·· ·· ·· ·· ·· ██ ██ ██ ··
P9    ·· ·· ·· ·· ·· ·· ·· ██ ██ ··
P10   ·· ·· ·· ·· ·· ·· ·· ·· ██ ██
Mốc   M0 ·· ·· M1 ·· M2 ·· M3 M4 M5
```

### 7.2 Mốc kiểm soát (milestone) và tiêu chí

| Mốc | Cuối tuần | Level | Tiêu chí đạt |
|---|---|---|---|
| **M0** | 1 | — | Repo + skeleton chạy; dữ liệu chốt; phân công; kế hoạch GT |
| **M1** | 4 | 1 | Detection + overlay vạch sân trên clip dev (video + hình); so sánh Canny vs top-hat |
| **M2** | 6 | 2 | Tracking + homography + top-down + heatmap (video demo 30–60 s); số liệu v0 |
| **M3** | 8 | 3 | Nhãn đội + team shape + quãng đường/tốc độ; bảng đánh giá v1 |
| **M4** | 9 | 3 | Ablation xong; số liệu đóng băng; báo cáo v1; **code freeze** (chỉ sửa lỗi) |
| **M5** | 10 | 3 (+4 nếu kịp) | Nộp: báo cáo PDF, repo public, demo, slide |

---

## 8. Kế hoạch đánh giá & tối ưu

### 8.1 Chỉ số theo từng thành phần

| Thành phần | Chỉ số | Ground truth cần | Công cụ | Ghi chú |
|---|---|---|---|---|
| Mask sân | IoU với mask vẽ tay | 10–20 mask | NumPy | tuỳ chọn |
| Đường / keypoint | tỉ lệ frame có ≥ 4 landmark đúng; sai số keypoint (px) | landmark click tay | script | báo cáo theo từng shot |
| Homography | reproj error (**mét**): mean/median/p95; line-overlay score; jitter của 4 điểm tham chiếu (px/frame) | 30–50 frame, landmark held-out | script | pixel phụ thuộc khoảng cách tới camera → ưu tiên mét (Phụ lục A6) |
| Detection | precision/recall, mAP@0.5, mAP@0.5:0.95 theo lớp; FPS | 100–200 frame có bbox | Ultralytics val / pycocotools | tách riêng lớp người và bóng |
| Tracking | MOTA, IDF1, HOTA, IDSW, Frag | đoạn 10–20 s dạng MOT | py-motmetrics / TrackEval | MOTA thiên về detection, IDF1 về nhất quán ID |
| Đội / vai trò | accuracy, confusion matrix | nhãn đội trên subset | scikit-learn | |
| Vị trí top-down | sai số vị trí (m) tại điểm biết trước; độ phân tán vị trí của người đứng yên | điểm/landmark biết trước | script | |
| Tốc độ / quãng đường | phân bố $v$; tỉ lệ mẫu > 12 m/s; (nếu có GT từ dataset) MAE | tuỳ nguồn | script | sanity check là chính |
| Hệ thống | FPS end-to-end; ms/frame từng stage; VRAM | — | `perf_counter` / profiler | ghi rõ phần cứng |

### 8.2 Các ablation (so sánh phương pháp)

Mỗi ablation phải có: **giả thuyết → biến thay đổi → metric → kết luận (kể cả khi giả thuyết sai).**

| ID | Câu hỏi | Biến thay đổi | Metric |
|---|---|---|---|
| A1 | Top-hat có lấy vạch tốt hơn Canny thô không? | Canny vs top-hat (Phase 2) | tỉ lệ đường đúng/sai; tỉ lệ frame ổn định |
| A2 | Cách khởi tạo/duy trì $H$ nào ổn nhất? | bán tự động + lan truyền vs tự động; có/không re-anchor | reproj error (m); tỉ lệ `valid`; drift theo thời gian |
| A3 | Làm mượt $H$ có giúp không? | không / làm mượt phần tử ma trận / làm mượt 4 điểm tham chiếu | jitter; reproj error |
| A4 | Kích cỡ model và `imgsz` | nano/small/medium × `imgsz` | mAP; FPS |
| A5 | Tracker nào tốt hơn? | SORT-tự-cài vs ByteTrack (vs DeepSORT) | IDF1, MOTA, HOTA, IDSW |
| A6 | Tracking trong không gian nào? | image-space vs pitch-space | IDF1, IDSW |
| A7 | Làm mượt quỹ đạo | không / moving average / Savitzky–Golay | phân bố vận tốc; tỉ lệ cờ lỗi; "quãng đường" của người đứng yên |
| A8 | Phân loại đội | KMeans màu vs embedding | accuracy |

### 8.3 Quy tắc đánh giá

- Không tuning trên test; mọi siêu tham số chọn trên dev; test chạy một lần ở Phase 9.
- Với sai số hình học, báo cáo cả **median và p95** (không chỉ mean — một vài frame hỏng sẽ kéo mean).
- Cố định seed; ghi phiên bản thư viện, trọng số, phần cứng.
- Mỗi bảng kèm 1–2 câu *diễn giải* (vì sao số như vậy), không chỉ dán số.
- Đưa cả kết quả xấu và ca lỗi vào báo cáo — đây là phần thể hiện năng lực phân tích mà đề đòi hỏi.

### 8.4 Tối ưu tốc độ & độ ổn định

- **Detector:** model nhỏ hơn hoặc `imgsz` thấp hơn (đo mAP–FPS); FP16; xuất ONNX/TensorRT; batch inference; chạy detector mỗi $k$ frame và để Kalman dự đoán ở giữa (đánh đổi độ chính xác).
- **I/O:** decode đa luồng/prefetch; ghi video bất đồng bộ.
- **Homography:** giới hạn số feature, dùng ROI, chỉ re-anchor khi điểm overlay thấp.
- **Ổn định:** làm mượt $H$ (giảm jitter); hysteresis cho nhãn đội; tracklet stitching (giảm IDSW).
- **Đo:** `time.perf_counter()` theo stage; báo cáo ms/frame từng stage + FPS tổng + phần cứng.

---

## 9. Báo cáo PDF & repo

### 9.1 Outline báo cáo (điều chỉnh theo hướng dẫn chung của môn)

| § | Nội dung | Gợi ý độ dài |
|---|---|---|
| 1 | Giới thiệu: bài toán, động lực, phạm vi (Level), đóng góp | 0.5–1 trang |
| 2 | Dữ liệu: nguồn, giấy phép, mô tả (bảng), tiền xử lý (ảnh before/after), GT và cách gán nhãn | 1–1.5 trang |
| 3 | Pipeline tổng thể: sơ đồ khối, data flow, quy ước toạ độ | 1 trang |
| 4 | Phương pháp: 4.1 tiền xử lý & mask · 4.2 biên/đường/keypoints · 4.3 homography & căn chỉnh · 4.4 detection · 4.5 tracking · 4.6 đội/bóng · 4.7 top-down & phân tích. Mỗi mục: ý tưởng → công thức chính → code chính → **Vai trò–Giả định–Hạn chế** | 5–8 trang |
| 5 | Thực nghiệm & thảo luận: thiết lập, chỉ số, bảng kết quả, ablation, định tính (input vs output, theo thời gian), tốc độ, phân tích lỗi | 3–5 trang |
| 6 | Phân công & đóng góp (bảng mục 6.4) | 0.5 trang |
| 7 | Kết luận & hướng phát triển | 0.5 trang |
| — | Tài liệu tham khảo; phụ lục (link repo, hướng dẫn chạy, bảng siêu tham số) | — |

Chỉ chèn các đoạn code chính vào báo cáo; code đầy đủ ở repo public (đề yêu cầu).

### 9.2 Bộ ba Vai trò – Giả định – Hạn chế (điền sẵn bản nháp)

| Module | Vai trò | Giả định | Hạn chế |
|---|---|---|---|
| Canny + Hough | Trích biên và đường vạch sân | Vạch trắng tương phản với cỏ; vạch xấp xỉ thẳng trong ảnh (méo ống kính nhỏ) | Sọc cỏ, bóng đổ, che khuất, zoom; vòng tròn thành ellipse |
| Homography (DLT + RANSAC) | Ánh xạ ảnh → sân (mét) | Điểm nằm trên mặt phẳng; mô hình pinhole; sân phẳng; kích thước sân biết trước | Bóng/người rời mặt sân; sân dốc; méo ống kính; sân thật khác 105×68 m |
| Căn chỉnh frame↔frame | Lan truyền $H$ theo thời gian | Camera chủ yếu xoay; đủ đặc trưng nền tĩnh | Drift tích luỹ; đặc trưng động (người, biển LED) gây nhiễu; zoom nhanh |
| YOLO pretrained | Phát hiện người/bóng | Miền ảnh gần COCO; đối tượng đủ lớn | Cầu thủ nhỏ/xa, che khuất; bóng nhỏ; không phân biệt vai trò |
| ByteTrack / SORT | Gán ID theo thời gian | Chuyển động gần tuyến tính ngắn hạn; detection đủ ổn định | ID switch khi chen nhau, áo giống màu, camera đổi mạnh; track đứt khi ra khỏi khung |
| Phân loại đội (KMeans màu) | Gán đội/vai trò | Hai đội áo khác màu; thân trên nhìn thấy | Áo gần màu/sọc; trọng tài/GK làm nhiễu cụm; điều kiện sáng |
| Top-down & phân tích | Số liệu chiến thuật | $H$ đúng; điểm chân ≈ điểm chạm sân | Chỉ phần nhìn thấy; sai số vị trí → sai số vận tốc; bbox cắt/che chân |

### 9.3 Cấu trúc repo đề xuất

```text
football-vision/
├── README.md                  # mục tiêu, GIF demo, cài đặt, cách chạy, kết quả, thành viên
├── requirements.txt           # hoặc environment.yml
├── configs/                   # YAML cho từng stage/thí nghiệm
├── data/                      # .gitignore (raw/, frames/, annotations/) + data/README.md
├── src/
│   ├── preprocessing/         # decode, resize, màu, shot detection, mask sân
│   ├── pitch/                 # template sân, Canny/Hough, keypoints
│   ├── homography/            # DLT, RANSAC, lan truyền, làm mượt, kiểm định
│   ├── detection/             # YOLO inference/fine-tune, hậu xử lý
│   ├── tracking/              # SORT tự cài, wrapper ByteTrack, stitching
│   ├── team/                  # phân loại đội/vai trò, possession
│   ├── analytics/             # top-down, heatmap, shape, tốc độ
│   ├── viz/                   # overlay, renderer video, hình báo cáo
│   └── eval/                  # metric, ablation
├── scripts/                   # run_pipeline.py, eval_*.py, make_figures.py
├── demo/                      # streamlit_app.py
├── notebooks/                 # khám phá, không chứa logic chính
├── reports/                   # figures/, tables/, report.pdf
└── tests/                     # test nhỏ cho hàm hình học (H, chiếu điểm, metric)
```

### 9.4 Checklist README

- [ ] Mô tả bài toán 3–5 dòng + GIF demo
- [ ] Cài đặt (môi trường, GPU/CPU) và lệnh chạy một dòng cho pipeline + demo
- [ ] Cách lấy dữ liệu và trọng số (link/script), ghi nguồn & giấy phép
- [ ] Bảng kết quả chính + link báo cáo
- [ ] Cấu trúc thư mục; bảng thành viên – phần phụ trách
- [ ] License và trích dẫn

---

## 10. Rủi ro & phương án dự phòng

| # | Rủi ro | Dấu hiệu sớm | Giảm thiểu | Plan B |
|---|---|---|---|---|
| 1 | Hough/đường sân không ổn | overlay "bơi", ít đường đúng | mask + top-hat + lọc theo hướng/độ dài | bán tự động + lan truyền; chọn clip dễ hơn |
| 2 | Camera zoom/pan mạnh, cut/replay | reproj error vọt lên khi zoom | tách shot; SIFT/AKAZE; re-anchor | dùng clip camera cố định làm clip chính |
| 3 | Vướng bản quyền dữ liệu | không rõ giấy phép | dùng dataset công khai / tự quay; ghi nguồn | clip ngắn mục đích học thuật + link (không đưa video thô vào repo) |
| 4 | Bóng khó phát hiện | recall bóng thấp | SAHI/tiling, nội suy | bỏ Level 4 (không ảnh hưởng phần lõi) |
| 5 | Nhiều ID switch | IDSW cao, quỹ đạo vỡ | tracklet stitching; pitch-space tracking; appearance | chấp nhận và phân tích lỗi kỹ |
| 6 | Thiếu GPU | YOLO chậm, hết quota | Colab/Kaggle; model nhỏ; hạ `imgsz`; cache | chạy offline, ghi FPS thật |
| 7 | Thiếu ground truth | không đủ nhãn cho mAP/IDF1 | lập kế hoạch GT từ Phase 1; pre-annotate | dùng dataset công khai nhỏ; nêu giới hạn thống kê |
| 8 | Thành viên bận / đóng góp lệch | PR thưa, task trễ | backup owner; board; check-in hàng tuần | tái phân bổ sớm; ghi nhận trong bảng đóng góp |
| 9 | Scope creep | thêm tính năng khi phần lõi chưa ổn | khoá Level; thay đổi phải qua họp nhóm | cắt Level 4/tuỳ chọn trước |
| 10 | Tích hợp muộn | module chạy riêng nhưng ghép thì lỗi | walking skeleton; integration run hàng tuần | đóng băng interface, dùng dữ liệu giả |

---

## 11. Checklist nộp bài

**Code & repo**

- [ ] Repo public; clone sạch chạy được theo README (do người không viết pipeline kiểm tra)
- [ ] `requirements.txt`/`environment.yml`, config YAML, seed cố định
- [ ] Script tái lập mọi bảng/hình (`scripts/eval_*.py`)
- [ ] Không có video thô hay khoá bí mật trong repo; có link tải dữ liệu/trọng số
- [ ] Tag `v1.0`; có license

**Báo cáo PDF**

- [ ] Mô tả bài toán & dữ liệu (nguồn, giấy phép, GT)
- [ ] Pipeline tổng thể có sơ đồ
- [ ] Mỗi module có công thức chính, code chính, Vai trò–Giả định–Hạn chế
- [ ] Hình input vs output; kết quả video theo trình tự thời gian
- [ ] Bảng định lượng + ablation + phân tích lỗi
- [ ] Bảng phân công & đóng góp từng thành viên
- [ ] Tài liệu tham khảo; trích dẫn nguồn dữ liệu/mô hình

**Demo & bảo vệ**

- [ ] Video demo 1–2 phút + demo chạy được (hoặc video dự phòng)
- [ ] Slide; mỗi người trình bày phần mình làm
- [ ] Đã diễn tập Q&A (mục 12)

---

## 12. Câu hỏi bảo vệ dự kiến (tự kiểm tra)

Mỗi câu kèm chỗ để ôn trong tài liệu này. Mục tiêu: trả lời được *từ nguyên lý*, không chỉ "thư viện làm giúp".

1. **Phần CV truyền thống của dự án nằm ở đâu?** → Canny/Hough, mask màu, homography (DLT), căn chỉnh frame, SORT tự cài (Phase 1–3, 5).
2. **Vì sao homography cần ≥ 4 điểm?** → 8 bậc tự do, mỗi điểm cho 2 phương trình độc lập (A4).
3. **Vì sao dùng điểm chân thay vì tâm bbox?** → giả định mặt phẳng (A3, Phase 7).
4. **Homography có đúng cho quả bóng đang bay không?** → Không: điểm nằm ngoài mặt phẳng sân (A3).
5. **RANSAC cần bao nhiêu vòng lặp?** → công thức ở A4 (ví dụ ≈ 72 khi 50% outlier, ≈ 570 khi 70%).
6. **Vì sao Hough dùng $(\rho,\theta)$ thay vì $(m,c)$?** → A1.
7. **Vì sao phải chuẩn hoá toạ độ trước DLT?** → điều kiện số của $A^\top A$ (A4).
8. **ByteTrack khác SORT/DeepSORT chỗ nào?** → vòng ghép thứ hai với detection điểm thấp; SORT chỉ IoU + Kalman; DeepSORT thêm appearance (A5).
9. **Vì sao tracking trong toạ độ sân có thể tốt hơn?** → khử chuyển động camera khỏi mô hình vận tốc đều; kiểm chứng bằng A6 (Phase 5).
10. **MOTA, IDF1, HOTA khác nhau gì?** → A6.
11. **Sai số vị trí 0.3 m ảnh hưởng thế nào đến vận tốc?** → khuếch đại thành cỡ 10 m/s nếu không làm mượt (A7).
12. **Làm sao biết homography đang drift?** → line-overlay score; reproj error trên landmark held-out (Phase 3).
13. **Vì sao đo sai số homography bằng mét thay vì pixel?** → A6.
14. **YOLO pretrain COCO có biết "thủ môn" không?** → Không; cách gán vai trò ở Phase 6.
15. **Hạn chế lớn nhất của hệ thống? Có thêm một tháng sẽ cải thiện gì?** → chuẩn bị 2–3 hạn chế có số đo + hướng khắc phục.
16. **Đóng góp của từng thành viên?** → bảng mục 6.4 + lịch sử commit/PR.

---

# Phụ lục A: Nền tảng toán (suy diễn từ đầu)

## A1. Hough transform cho đường thẳng

Một đường thẳng trong ảnh viết được dưới dạng

$$x\cos\theta+y\sin\theta=\rho,\qquad \theta\in[0,\pi),\ \rho\in\mathbb{R},$$

trong đó $(\cos\theta,\sin\theta)$ là pháp tuyến đơn vị và $\rho$ là khoảng cách có dấu từ gốc toạ độ tới đường.

**Vì sao không dùng $y=mx+c$?** Đường thẳng đứng có $m\to\infty$; không gian tham số không bị chặn nên khó rời rạc hoá. Dạng $(\rho,\theta)$ thì bị chặn.

**Vì sao bỏ phiếu hoạt động?** Một điểm biên $(x_i,y_i)$ nằm trên mọi đường thoả $\rho=x_i\cos\theta+y_i\sin\theta$. Đặt $x_i=r_i\cos\varphi_i,\ y_i=r_i\sin\varphi_i$ thì

$$\rho(\theta)=r_i\cos(\theta-\varphi_i),$$

tức một đường hình sin trong mặt phẳng $(\theta,\rho)$. Nếu nhiều điểm biên cùng nằm trên đường $(\rho_0,\theta_0)$ thì mọi đường hình sin của chúng đều đi qua điểm $(\theta_0,\rho_0)$, nên ô tích luỹ tại đó nhận nhiều phiếu nhất.

**Thuật toán.** Bảng tích luỹ $A[\rho,\theta]$; mỗi điểm biên cộng phiếu cho mọi $\theta$ rời rạc; cực đại cục bộ vượt ngưỡng là đường thẳng. Độ phức tạp $O(N_{\text{biên}}\cdot N_\theta)$. Probabilistic Hough chỉ lấy mẫu một phần điểm biên và trả về *đoạn* thẳng.

## A2. Đường thẳng, giao điểm và toạ độ đồng nhất

Đường $ax+by+c=0$ ↔ vector $\mathbf{l}=(a,b,c)^\top$; điểm $(x,y)$ ↔ $\tilde{\mathbf{p}}=(x,y,1)^\top$; điểm thuộc đường ⇔ $\mathbf{l}^\top\tilde{\mathbf{p}}=0$.

**Giao điểm hai đường:** $\tilde{\mathbf{p}}=\mathbf{l}_1\times\mathbf{l}_2$.
*Chứng minh.* $\mathbf{l}_1^\top(\mathbf{l}_1\times\mathbf{l}_2)=\det[\mathbf{l}_1,\mathbf{l}_1,\mathbf{l}_2]=0$ và tương tự $\mathbf{l}_2^\top(\mathbf{l}_1\times\mathbf{l}_2)=0$ (tích hỗn tạp có hai vector trùng nhau bằng 0), nên $\tilde{\mathbf{p}}$ thuộc cả hai đường. Chia cho thành phần thứ ba để về $(x,y)$. $\square$

**Đường qua hai điểm:** $\mathbf{l}=\tilde{\mathbf{p}}_1\times\tilde{\mathbf{p}}_2$ (chứng minh đối ngẫu). Đường từ Hough: $\mathbf{l}=(\cos\theta,\sin\theta,-\rho)^\top$.

## A3. Homography từ mặt phẳng và từ phép xoay thuần

**(a) Homography cảm sinh bởi mặt phẳng.** Camera pinhole: $\lambda\tilde{\mathbf{x}}=K[R\,|\,\mathbf{t}]\tilde{\mathbf{X}}$ với $\tilde{\mathbf{X}}=(X,Y,Z,1)^\top$. Chọn hệ toạ độ thế giới sao cho mặt sân là $Z=0$, khi đó $\tilde{\mathbf{X}}=(X,Y,0,1)^\top$ và cột thứ ba của $R$ bị nhân với 0:

$$\lambda\tilde{\mathbf{x}}=K[\mathbf{r}_1\ \ \mathbf{r}_2\ \ \mathbf{t}]\begin{pmatrix}X\\Y\\1\end{pmatrix}\quad\Rightarrow\quad H_{\text{sân}\to\text{ảnh}}=K[\mathbf{r}_1\ \ \mathbf{r}_2\ \ \mathbf{t}].$$

$H$ khả nghịch khi mặt phẳng không đi qua tâm camera; $H_{\text{ảnh}\to\text{sân}}=H^{-1}$. Ma trận $3\times3$ xác định sai khác một hệ số tỉ lệ nên có **8 bậc tự do**.

**Hệ quả quan trọng:** công thức chỉ đúng cho điểm có $Z=0$. Điểm ở độ cao $h>0$ (tâm bbox, quả bóng bay) bị ánh xạ lệch dọc theo tia nhìn — vì vậy dùng *điểm chân*.

**(b) Homography giữa hai frame khi camera chỉ xoay.** Đặt gốc thế giới tại tâm quang (không có tịnh tiến): $\lambda_1\tilde{\mathbf{x}}_1=K_1R_1\mathbf{X}$ và $\lambda_2\tilde{\mathbf{x}}_2=K_2R_2\mathbf{X}$. Từ phương trình đầu, $\mathbf{X}=\lambda_1R_1^{-1}K_1^{-1}\tilde{\mathbf{x}}_1$; thế vào phương trình sau:

$$\tilde{\mathbf{x}}_2\sim K_2R_2R_1^{-1}K_1^{-1}\,\tilde{\mathbf{x}}_1=:G\,\tilde{\mathbf{x}}_1 .$$

$G$ không chứa độ sâu của điểm cảnh ⇒ đúng cho **mọi** điểm (kể cả khán đài); zoom chỉ đổi $K$ nên vẫn là homography. Camera truyền hình trên chân đứng xấp xỉ điều kiện này, nhưng không tuyệt đối (méo ống kính, tâm quang dịch nhẹ khi zoom).

**Ghép chuỗi:** nếu $G_t$ ánh xạ frame $t\to t-1$ và $H_{t-1}$ ánh xạ frame $t-1\to$ sân thì $H_t=H_{t-1}G_t$ (sai khác tỉ lệ). Sai số của từng $G_t$ nhân dồn ⇒ drift.

## A4. Ước lượng homography: DLT, chuẩn hoá, RANSAC, tinh chỉnh

**Thiết lập.** Cặp tương ứng $\tilde{\mathbf{x}}_i=(x_i,y_i,1)^\top$ (ảnh) và $\tilde{\mathbf{X}}_i=(X_i,Y_i,1)^\top$ (sân). Cần $H$ sao cho $\tilde{\mathbf{X}}_i\sim H\tilde{\mathbf{x}}_i$ (bằng nhau sai khác hệ số). Điều này tương đương

$$\tilde{\mathbf{X}}_i\times(H\tilde{\mathbf{x}}_i)=\mathbf{0},$$

vì hai vector song song khi và chỉ khi tích có hướng bằng 0; cách viết này khử hệ số tỉ lệ chưa biết mà không cần chia.

**Khai triển.** Gọi $\mathbf{h}_1^\top,\mathbf{h}_2^\top,\mathbf{h}_3^\top$ là ba hàng của $H$ và $\mathbf{h}=(\mathbf{h}_1;\mathbf{h}_2;\mathbf{h}_3)\in\mathbb{R}^9$. Khi đó

$$\tilde{\mathbf{X}}_i\times H\tilde{\mathbf{x}}_i=\begin{pmatrix}Y_i\,\mathbf{h}_3^\top\tilde{\mathbf{x}}_i-\mathbf{h}_2^\top\tilde{\mathbf{x}}_i\\[2pt]\mathbf{h}_1^\top\tilde{\mathbf{x}}_i-X_i\,\mathbf{h}_3^\top\tilde{\mathbf{x}}_i\\[2pt]X_i\,\mathbf{h}_2^\top\tilde{\mathbf{x}}_i-Y_i\,\mathbf{h}_1^\top\tilde{\mathbf{x}}_i\end{pmatrix}=\mathbf{0}.$$

Hàng thứ ba $=-(X_i\cdot\text{hàng 1}+Y_i\cdot\text{hàng 2})$ (kiểm tra: $X_i(Y_i\mathbf{h}_3^\top\tilde{\mathbf{x}}_i-\mathbf{h}_2^\top\tilde{\mathbf{x}}_i)+Y_i(\mathbf{h}_1^\top\tilde{\mathbf{x}}_i-X_i\mathbf{h}_3^\top\tilde{\mathbf{x}}_i)=Y_i\mathbf{h}_1^\top\tilde{\mathbf{x}}_i-X_i\mathbf{h}_2^\top\tilde{\mathbf{x}}_i$), nên mỗi cặp điểm chỉ cho **2 phương trình độc lập**:

$$\underbrace{\begin{pmatrix}\mathbf{0}^\top&-\tilde{\mathbf{x}}_i^\top&Y_i\tilde{\mathbf{x}}_i^\top\\ \tilde{\mathbf{x}}_i^\top&\mathbf{0}^\top&-X_i\tilde{\mathbf{x}}_i^\top\end{pmatrix}}_{A_i\ (2\times9)}\mathbf{h}=\mathbf{0}.$$

Xếp chồng $n$ cặp: $A\in\mathbb{R}^{2n\times9}$, $A\mathbf{h}=\mathbf{0}$.

**Đếm bậc tự do.** $\mathbf{h}$ có 9 ẩn, mất 1 vì tỉ lệ ⇒ 8 ẩn ⇒ $2n\ge8\Rightarrow n\ge4$. Cần không có ba điểm thẳng hàng (nếu có, $A$ mất hạng và nghiệm không duy nhất).

**Nghiệm.** Với $n=4$: không gian nghiệm 1 chiều (nghiệm chính xác). Với $n>4$ và có nhiễu: giải

$$\min_{\lVert\mathbf{h}\rVert=1}\lVert A\mathbf{h}\rVert^2=\min_{\lVert\mathbf{h}\rVert=1}\mathbf{h}^\top A^\top A\,\mathbf{h}.$$

$A^\top A$ đối xứng nửa xác định dương nên thương Rayleigh $\mathbf{h}^\top A^\top A\mathbf{h}/\mathbf{h}^\top\mathbf{h}\ge\lambda_{\min}(A^\top A)$, dấu bằng khi $\mathbf{h}$ là vector riêng tương ứng. Do đó $\mathbf{h}$ là **vector kỳ dị phải ứng với giá trị kỳ dị nhỏ nhất** của $A$: với $A=U\Sigma V^\top$, lấy $\mathbf{h}=V_{:,9}$. Reshape thành $H\in\mathbb{R}^{3\times3}$.

**Chuẩn hoá (Hartley).** Với mỗi tập điểm, tìm phép tịnh tiến + co giãn đồng đều $T$ sao cho trọng tâm về gốc và khoảng cách trung bình tới gốc bằng $\sqrt2$: $\hat{\mathbf{x}}_i=T\tilde{\mathbf{x}}_i,\ \hat{\mathbf{X}}_i=T'\tilde{\mathbf{X}}_i$. Giải DLT cho $\hat H$ rồi $H=T'^{-1}\hat HT$.
*Trực giác:* các phần tử của $A$ trộn các cỡ từ ~1 tới ~$10^5$–$10^6$ (toạ độ pixel nhân toạ độ mét) ⇒ $A^\top A$ điều kiện xấu và sai số đại số bị chi phối bởi điểm có toạ độ lớn; chuẩn hoá đưa mọi cột về cùng cỡ ~1 và làm nghiệm gần như không phụ thuộc cách chọn gốc/đơn vị toạ độ.

**RANSAC.** Lặp: chọn ngẫu nhiên 4 cặp → tính $H$ → đếm inlier (sai số chiếu < $\tau$) → giữ $H$ nhiều inlier nhất → tinh chỉnh trên toàn bộ inlier. Số vòng lặp để với xác suất ≥ $p$ có ít nhất một mẫu toàn inlier, khi tỉ lệ outlier là $\varepsilon$:

$$N=\frac{\ln(1-p)}{\ln\!\big(1-(1-\varepsilon)^4\big)}.$$

Ví dụ $p=0.99$: $\varepsilon=0.5\Rightarrow N=\ln0.01/\ln(1-0.0625)\approx72$; $\varepsilon=0.7\Rightarrow(0.3)^4=0.0081\Rightarrow N\approx570$. Số vòng lặp tăng rất nhanh theo $\varepsilon$ ⇒ lọc outlier trước (mask sân, lọc đường) là đáng giá.

**Tinh chỉnh phi tuyến.** Sau DLT/RANSAC, tối thiểu hoá sai số hình học

$$\min_H\sum_i\big\lVert\mathbf{X}_i-\pi(H\tilde{\mathbf{x}}_i)\big\rVert^2,\qquad \pi(u,v,w)=(u/w,\ v/w),$$

bằng Levenberg–Marquardt. Khác DLT ở chỗ đại lượng được tối thiểu có ý nghĩa vật lý (DLT tối thiểu sai số đại số $\lVert A\mathbf{h}\rVert$, không có ý nghĩa hình học).

## A5. Kalman filter và bài toán gán trong tracking

**Mô hình vận tốc đều trong toạ độ sân.** Trạng thái $\mathbf{x}_k=[X,Y,\dot X,\dot Y]^\top$:

$$\mathbf{x}_k=F\mathbf{x}_{k-1}+\mathbf{w},\quad \mathbf{z}_k=M\mathbf{x}_k+\mathbf{v},\quad \mathbf{w}\sim\mathcal{N}(0,Q),\ \mathbf{v}\sim\mathcal{N}(0,R),$$

$$F=\begin{pmatrix}1&0&\Delta t&0\\0&1&0&\Delta t\\0&0&1&0\\0&0&0&1\end{pmatrix},\qquad M=\begin{pmatrix}1&0&0&0\\0&1&0&0\end{pmatrix}.$$

(Dùng $M$ cho ma trận đo để không trùng với homography $H$.)

**Dự đoán:** $\hat{\mathbf{x}}^-=F\hat{\mathbf{x}}^+_{k-1},\quad P^-=FP^+_{k-1}F^\top+Q$.
**Cập nhật:** $S=MP^-M^\top+R,\quad K=P^-M^\top S^{-1},\quad \hat{\mathbf{x}}^+=\hat{\mathbf{x}}^-+K(\mathbf{z}-M\hat{\mathbf{x}}^-),\quad P^+=(I-KM)P^-$.

**Suy ra $K$.** Sai số hậu nghiệm $\mathbf{e}^+=(I-KM)\mathbf{e}^--K\mathbf{v}$; hai thành phần độc lập nên
$P^+=(I-KM)P^-(I-KM)^\top+KRK^\top$. Tối thiểu vết:

$$\frac{\partial\,\mathrm{tr}(P^+)}{\partial K}=-2(I-KM)P^-M^\top+2KR=0\ \Rightarrow\ K(R+MP^-M^\top)=P^-M^\top\ \Rightarrow\ K=P^-M^\top S^{-1}.$$

*Trực giác:* $R\to0$ (đo rất chắc) ⇒ tin hoàn toàn vào đo; $R\to\infty$ ⇒ $K\to0$, tin vào dự đoán.

**Vì sao toạ độ sân tốt hơn toạ độ ảnh cho mô hình vận tốc đều (giả thuyết cần kiểm chứng bằng A6).** Trong ảnh, vận tốc biểu kiến = vận tốc thật + chuyển động do camera pan/zoom; thành phần camera thay đổi theo thời gian nên vi phạm vận tốc đều và Kalman dự đoán lệch. Trong toạ độ sân, nếu $H_t$ đủ chính xác thì chuyển động camera đã được khử, còn lại chuyển động của cầu thủ (gia tốc bị chặn bởi giới hạn sinh lý).

**Bài toán gán.** Ma trận chi phí $C_{ij}=1-\mathrm{IoU}(\hat b_i,d_j)$ (hoặc khoảng cách trong toạ độ sân); giải $\min_\pi\sum_iC_{i\pi(i)}$ với $\pi$ đơn ánh bằng Hungarian ($O(n^3)$, `scipy.optimize.linear_sum_assignment`); loại các cặp có $C_{ij}$ vượt ngưỡng (gating). Trong toạ độ sân có thể dùng khoảng cách Mahalanobis $d^2_{ij}=(\mathbf{z}_j-M\hat{\mathbf{x}}_i)^\top S_i^{-1}(\mathbf{z}_j-M\hat{\mathbf{x}}_i)$; dưới mô hình Gauss $d^2\sim\chi^2_2$ nên cổng 95% là $5.99$.

**ByteTrack.** Chia detection theo điểm: cao ($\ge\tau_{\text{high}}$) và thấp ($\tau_{\text{low}}\le s<\tau_{\text{high}}$). Vòng 1: ghép track với detection điểm cao. Vòng 2: ghép các track *chưa ghép được* với detection điểm thấp (thường là đối tượng bị che nên điểm thấp). Chỉ detection điểm cao chưa ghép mới khởi tạo track mới. Track không được ghép giữ trong `track_buffer` frame rồi mới xoá.

## A6. Định nghĩa các chỉ số

- **IoU:** $\mathrm{IoU}(A,B)=\dfrac{|A\cap B|}{|A\cup B|}$.
- **AP:** diện tích dưới đường precision–recall, $\mathrm{AP}=\int_0^1p(r)\,dr$ (COCO nội suy 101 điểm); mAP = trung bình theo lớp; mAP@0.5:0.95 còn trung bình thêm theo ngưỡng IoU $0.50,0.55,\dots,0.95$.
- **MOTA:** $1-\dfrac{\mathrm{FN}+\mathrm{FP}+\mathrm{IDSW}}{\mathrm{GT}}$; có thể âm; chịu ảnh hưởng chủ yếu bởi chất lượng detection.
- **IDF1:** $\dfrac{2\,\mathrm{IDTP}}{2\,\mathrm{IDTP}+\mathrm{IDFP}+\mathrm{IDFN}}$; F1 ở mức *danh tính* — ghép tối ưu giữa quỹ đạo dự đoán và quỹ đạo GT trên toàn chuỗi, phạt việc đổi ID.
- **HOTA:** $\mathrm{HOTA}_\alpha=\sqrt{\mathrm{DetA}_\alpha\cdot\mathrm{AssA}_\alpha}$, trung bình trên $\alpha\in\{0.05,0.10,\dots,0.95\}$; $\mathrm{DetA}_\alpha=\dfrac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}+\mathrm{FP}}$; $\mathrm{AssA}_\alpha$ đo độ chính xác liên kết danh tính trên các TP. Cân bằng giữa detection và association.
- **Sai số homography:** $e=\dfrac1M\sum_{j=1}^M\lVert\pi(H\tilde{\mathbf{x}}_j)-\mathbf{X}_j\rVert$ (mét) trên landmark held-out. *Vì sao mét thay vì pixel:* một pixel ở vùng gần camera ứng với vài cm, ở vùng xa ứng với hàng decimét ⇒ sai số pixel không đồng nhất về vật lý; mét thì có.

## A7. Nhiễu vị trí khuếch đại thành nhiễu vận tốc

Giả sử vị trí đo $\mathbf{P}_t=\mathbf{P}^*_t+\boldsymbol\epsilon_t$, $\boldsymbol\epsilon_t\sim\mathcal{N}(0,\sigma^2I_2)$ **độc lập giữa các frame**. Khi đó $\Delta\mathbf{P}=\Delta\mathbf{P}^*+(\boldsymbol\epsilon_t-\boldsymbol\epsilon_{t-1})$ với nhiễu có hiệp phương sai $2\sigma^2I_2$, nên độ lệch chuẩn của vận tốc trên mỗi trục là

$$\frac{\sqrt2\,\sigma}{\Delta t}.$$

Ví dụ $\sigma=0.3$ m, 25 fps ($\Delta t=0.04$ s): $\sqrt2\cdot0.3/0.04\approx10.6$ m/s trên mỗi trục — lớn hơn cả tốc độ chạy nước rút.

**Quãng đường bị phồng.** Độ lớn của nhiễu dịch chuyển có phân phối Rayleigh với tham số $s=\sqrt2\sigma$, kỳ vọng $s\sqrt{\pi/2}=\sigma\sqrt\pi\approx0.53$ m/frame. Một cầu thủ *đứng yên* sẽ "chạy" $\approx0.53\times25\times60\approx800$ m mỗi phút. ⇒ Bắt buộc làm mượt trước khi tính vận tốc và quãng đường.

**Savitzky–Golay.** Khớp đa thức bậc $p$ trong cửa sổ $w$ điểm (lẻ) bằng bình phương tối thiểu, rồi lấy đạo hàm giải tích của đa thức tại tâm cửa sổ. Chọn cửa sổ ≈ 0.5–1 s ($w\approx13$–25 ở 25 fps), $p=2$–3. Đánh đổi: cửa sổ lớn giảm nhiễu nhưng gây lệch ở chỗ gia tốc đột ngột. Giới hạn vật lý để kiểm tra: tốc độ đỉnh của cầu thủ ≈ 10 m/s ⇒ mẫu > 12 m/s coi là lỗi (đổi ID, homography nhảy).
*Lưu ý:* lỗi tương quan theo thời gian (ví dụ sai số $H$ kéo dài nhiều frame) *không* bị làm mượt loại bỏ.

## A8. Heatmap và team shape

**Heatmap (KDE Gauss).** Với các vị trí $\{\mathbf{P}_j\}_{j=1}^N$ trên sân:

$$\hat f(\mathbf{x})=\frac1N\sum_{j=1}^N\frac{1}{2\pi\sigma^2}\exp\!\Big(-\frac{\lVert\mathbf{x}-\mathbf{P}_j\rVert^2}{2\sigma^2}\Big).$$

Thực hành: histogram 2D rồi lọc Gaussian (xấp xỉ KDE, rẻ hơn). Chuẩn hoá theo *thời gian nhìn thấy* — nếu không, cầu thủ ở trong khung lâu hơn sẽ "nóng" hơn chỉ vì được quay nhiều. Chọn $\sigma$ cỡ sai số vị trí (vài mét): nhỏ hơn thì heatmap lốm đốm theo nhiễu, lớn hơn thì mờ mất cấu trúc.

**Team shape.** Với $N$ cầu thủ trường (có thể loại GK), vị trí $\mathbf{P}_i=(X_i,Y_i)$:

- Centroid: $\mathbf{C}=\dfrac1N\sum_i\mathbf{P}_i$.
- Độ dài: $L=\max_iX_i-\min_iX_i$; độ rộng: $W=\max_iY_i-\min_iY_i$.
- Stretch index: $S=\dfrac1N\sum_i\lVert\mathbf{P}_i-\mathbf{C}\rVert$.
- Diện tích convex hull: `scipy.spatial.ConvexHull(...).volume` (trong 2D, `volume` là diện tích, `area` là chu vi).

$L$ và $W$ nhạy với outlier (một cầu thủ đứng xa) ⇒ báo cáo thêm $S$ và hull. Hull cần ≥ 3 điểm không thẳng hàng.

---

# Phụ lục B: Template sân (mét)

Hệ toạ độ: gốc ở góc trên-trái, $X$ dọc chiều dài (0–105), $Y$ dọc chiều rộng (0–68). Kích thước 105×68 m là chuẩn thường dùng; sân thật có thể khác — nếu tự quay, hãy đo.

Các kích thước theo Luật bóng đá: khu phạt đền sâu 16.5 m, rộng $7.32+2\times16.5=40.32$ m; khu 5m50 sâu 5.5 m, rộng $7.32+2\times5.5=18.32$ m; chấm phạt đền cách đường biên ngang 11 m; bán kính vòng tròn giữa sân và cung tròn khu phạt đền 9.15 m.

| Landmark | $(X,Y)$ |
|---|---|
| Bốn góc sân | $(0,0)$, $(105,0)$, $(105,68)$, $(0,68)$ |
| Hai đầu vạch giữa sân | $(52.5,0)$, $(52.5,68)$ |
| Điểm giữa sân | $(52.5,34)$ |
| Khu phạt đền trái: giao với đường biên ngang | $(0,13.84)$, $(0,54.16)$ |
| Khu phạt đền trái: góc ngoài | $(16.5,13.84)$, $(16.5,54.16)$ |
| Khu 5m50 trái: giao với đường biên ngang | $(0,24.84)$, $(0,43.16)$ |
| Khu 5m50 trái: góc ngoài | $(5.5,24.84)$, $(5.5,43.16)$ |
| Chấm phạt đền trái | $(11,34)$ |
| Phía phải | đối xứng: $X\mapsto105-X$ |

Trong đó $13.84=(68-40.32)/2$, $54.16=(68+40.32)/2$, $24.84=(68-18.32)/2$, $43.16=(68+18.32)/2$.

```python
# pitch_template.py — mét; gốc trên-trái; X dọc chiều dài, Y dọc chiều rộng
L, W = 105.0, 68.0
PA_D, PA_W = 16.5, 40.32          # khu phạt đền: sâu x rộng
GA_D, GA_W = 5.5, 18.32           # khu 5m50
pa_y0, pa_y1 = (W - PA_W) / 2, (W + PA_W) / 2   # 13.84, 54.16
ga_y0, ga_y1 = (W - GA_W) / 2, (W + GA_W) / 2   # 24.84, 43.16

CORNERS = {"tl": (0, 0), "tr": (L, 0), "br": (L, W), "bl": (0, W)}
HALFWAY = {"top": (L / 2, 0), "bottom": (L / 2, W), "center": (L / 2, W / 2)}
LEFT = {
    "pa_goalline_top": (0, pa_y0),   "pa_goalline_bottom": (0, pa_y1),
    "pa_outer_top": (PA_D, pa_y0),   "pa_outer_bottom": (PA_D, pa_y1),
    "ga_goalline_top": (0, ga_y0),   "ga_goalline_bottom": (0, ga_y1),
    "ga_outer_top": (GA_D, ga_y0),   "ga_outer_bottom": (GA_D, ga_y1),
    "penalty_spot": (11.0, W / 2),
}
RIGHT = {k: (L - x, y) for k, (x, y) in LEFT.items()}
```

---

# Phụ lục C: Đoạn code mẫu (xương sống, không phải bản hoàn chỉnh)

```python
import cv2, numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter

# 1) Homography: ước lượng sân -> ảnh để ngưỡng RANSAC tính bằng PIXEL, rồi nghịch đảo.
#    (cv2.findHomography: ngưỡng ransacReprojThreshold theo đơn vị của tập ĐÍCH)
H_p2i, inliers = cv2.findHomography(pitch_pts, img_pts, cv2.RANSAC, 3.0)  # pitch(m) -> pixel
H_i2p = np.linalg.inv(H_p2i); H_i2p /= H_i2p[2, 2]                        # ảnh -> sân (m)

# 2) Chiếu điểm chân cầu thủ (giữa cạnh dưới bbox) sang toạ độ sân
feet = np.float32([[(x1 + x2) / 2, y2] for x1, y1, x2, y2 in boxes]).reshape(-1, 1, 2)
XY = cv2.perspectiveTransform(feet, H_i2p).reshape(-1, 2)                 # (X, Y) mét

# 3) Lan truyền homography theo chuỗi: G_t ánh xạ frame t -> frame t-1
G_t, _ = cv2.findHomography(pts_t, pts_tm1, cv2.RANSAC, 3.0)
H_t = H_prev @ G_t; H_t /= H_t[2, 2]

# 4) Làm mượt H: làm mượt vị trí ảnh của 4 điểm tham chiếu rồi dựng lại bằng 4 điểm
#    H_t = cv2.getPerspectiveTransform(smoothed_img_pts, ref_pitch_pts)

# 5) Heatmap trên sân (hàng = Y, cột = X) rồi làm mờ Gauss
hist, _, _ = np.histogram2d(Y, X, bins=[68, 105], range=[[0, 68], [0, 105]])
heat = gaussian_filter(hist, sigma=2.0)

# 6) Vận tốc SAU khi làm mượt (đạo hàm giải tích của đa thức Savitzky–Golay)
v = savgol_filter(P, window_length=15, polyorder=2, deriv=1, delta=1 / fps, axis=0)  # m/s
speed = np.linalg.norm(v, axis=1)
```

---

# Phụ lục D: Tài liệu tham khảo gợi ý

**Hình học & ước lượng**
- J. Canny (1986). *A Computational Approach to Edge Detection.* IEEE TPAMI.
- R. O. Duda, P. E. Hart (1972). *Use of the Hough Transformation to Detect Lines and Curves in Pictures.* CACM.
- R. Hartley, A. Zisserman (2004). *Multiple View Geometry in Computer Vision*, 2nd ed. — Chương 4 (DLT, chuẩn hoá, RANSAC).
- M. A. Fischler, R. C. Bolles (1981). *Random Sample Consensus.* CACM.

**Detection & tracking**
- J. Redmon et al. (2016). *You Only Look Once.* CVPR; và tài liệu Ultralytics cho phiên bản dùng.
- A. Bewley et al. (2016). *Simple Online and Realtime Tracking (SORT).* ICIP.
- N. Wojke et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric (DeepSORT).* ICIP.
- Y. Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box.* ECCV.
- R. E. Kalman (1960). *A New Approach to Linear Filtering and Prediction Problems.*

**Chỉ số**
- K. Bernardin, R. Stiefelhagen (2008). *Evaluating Multiple Object Tracking Performance: The CLEAR MOT Metrics.* (MOTA)
- E. Ristani et al. (2016). *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking.* (IDF1)
- J. Luiten et al. (2021). *HOTA: A Higher Order Metric for Evaluating Multi-Object Tracking.* IJCV.

**Dữ liệu & làm mượt**
- SoccerNet (Giancola et al., 2018) và các benchmark calibration/tracking đi kèm — đọc trang chính thức về điều khoản.
- SportsMOT (Cui et al., 2023). ICCV.
- A. Savitzky, M. J. E. Golay (1964). *Smoothing and Differentiation of Data by Simplified Least Squares Procedures.*
