# Báo cáo dự án FACE — End-to-End

**Repository:** [https://github.com/ITIS-mphuong169/FACE](https://github.com/ITIS-mphuong169/FACE)

**Phiên bản tài liệu:** 2026-06-04

---

## Mục lục

1. [Tóm tắt](#1-tóm-tắt)
2. [Bài toán và mục tiêu](#2-bài-toán-và-mục-tiêu)
3. [Kiến trúc end-to-end](#3-kiến-trúc-end-to-end)
4. [Cấu trúc repository](#4-cấu-trúc-repository)
5. [Thuật toán FACE](#5-thuật-toán-face)
6. [Hai case sử dụng](#6-hai-case-sử-dụng)
7. [Định dạng dữ liệu](#7-định-dạng-dữ-liệu)
8. [Cách cài đặt](#8-cách-cài-đặt)
9. [Hướng dẫn chạy từng pipeline](#9-hướng-dẫn-chạy-từng-pipeline)
10. [Kết quả đầu ra](#10-kết-quả-đầu-ra)
11. [Lưu ý Git và dữ liệu lớn](#11-lưu-ý-git-và-dữ-liệu-lớn)
12. [Xử lý sự cố](#12-xử-lý-sự-cố)
13. [Dataset `movenet_with_paths.csv` và chuẩn hóa CSV](#13-dataset-movenet_with_pathscsv-và-chuẩn-hóa-csv)
14. [Phụ lục — Tạo lại toàn bộ CSV từ đầu](#14-phụ-lục--tạo-lại-toàn-bộ-csv-từ-đầu)

---

## 1. Tóm tắt

Dự án **FACE** (*Feasible and Actionable Counterfactual Explanations*) triển khai pipeline sửa tư thế yoga dựa trên skeleton:

1. **Trích xuất feature** từ ảnh (MoveNet — 17 keypoint, 51 chiều).
2. **Phân loại** tư thế đúng/sai hoặc đánh giá độ tin cậy (MLP, ExtraTrees, SVM, …).
3. **Tìm counterfactual khả thi** bằng đồ thị FACE trên tập train (luôn trả về **ảnh thật** trong dataset).
4. **Trực quan hóa** kết quả (JSON, PCA, cặp ảnh wrong → suggested).

Prototype gồm ba phần chính:

| Thành phần | Vai trò |
|------------|---------|
| `YogaSinglePose-PTIT-main/` | Trích skeleton, train classifier cơ bản, demo webcam |
| `core_yoga_face/` | Pipeline FACE đầy đủ (train → demo → visualize) |
| Root (`run_yoga_label_target_tests.py`) | Thí nghiệm multiclass trên `poses_yoga_multiclass.csv`, không chuẩn hóa feature |

---

## 2. Bài toán và mục tiêu

### 2.1 Vấn đề

Người tập yoga cần biết:

- Động tác hiện tại có **đúng loại** không (tree, warrior, downdog, …)?
- Nếu model **không tin** (confidence thấp) hoặc **dự đoán sai**, nên **bắt chước tư thế nào**?

Giải pháp naive (trung bình vector pose, nội suy ảo) dễ tạo tư thế **không tồn tại** trong dữ liệu. FACE khắc phục bằng cách chọn mẫu **có sẵn** trong tập train và đi theo **đường đi ngắn** trên đồ thị pose, có phạt vùng mật độ thấp.

### 2.2 Đầu ra mong muốn

| Đầu vào | Đầu ra |
|---------|--------|
| Ảnh / skeleton CSV | Nhãn dự đoán + xác suất từng lớp |
| Mẫu cần sửa | `image_path` của ảnh gợi ý (counterfactual) |
| Phân tích | Đường đi FACE, biểu đồ PCA, cặp ảnh so sánh |

---

## 3. Kiến trúc end-to-end

```
┌─────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 0 — Thu thập & trích xuất (YogaSinglePose-PTIT-main)              │
│  Ảnh / webcam → MoveNet / BlazePose / MMPose → CSV skeleton             │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 1 — Chuẩn bị feature                                            │
│  Center mid-hip, scale torso/bbox → vector 51 (kp hoặc f0..f50)        │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 2 — Phân loại                                                   │
│  Train MLP / ExtraTrees / RF → predict + predict_proba                  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 3 — FACE graph search                                           │
│  kNN graph(train) + query node → Dijkstra → ảnh counterfactual thật    │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  BƯỚC 4 — Trình bày                                                   │
│  result.json, PCA, skeleton compare, cặp ảnh PNG                        │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.1 Bốn bước thuật toán (theo README gốc)

| Bước | Mô tả |
|------|--------|
| **1. Feature extraction** | MoveNet: 17 keypoint × (x, y, confidence) = 51 feature |
| **2. Classification** | MLP / ExtraTrees quyết định label và confidence |
| **3. FACE graph search** | Đồ thị kNN, trọng số cạnh theo mật độ KDE, shortest path |
| **4. Actionable feedback** | Ảnh mẫu + visualization (mở rộng: hướng dẫn ngôn ngữ tự nhiên) |

---

## 4. Cấu trúc repository

```
FACE/
├── README.md                          # Tổng quan pipeline root
├── report.md                          # Tài liệu này
├── requirements.txt                   # Dependencies root
├── poses_yoga_multiclass.csv          # Dataset feature sẵn (f0..f50)
├── run_yoga_label_target_tests.py     # Thí nghiệm label-target + FACE
├── render_counterfactual_image_pairs.py  # Vẽ cặp ảnh từ CSV output
├── main.tex                           # Tài liệu LaTeX (paper/report)
│
├── core_yoga_face/                    # Pipeline FACE chính
│   ├── README.md
│   ├── requirements.txt
│   ├── data/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   └── movenet_with_paths.csv
│   ├── src/
│   │   ├── prepare_data.py            # Tách train/test
│   │   ├── load_dataset.py            # Load + normalize skeleton
│   │   ├── train_classifier.py        # MLP / RF / MLP torch
│   │   ├── face_graph.py              # FACE graph + Dijkstra
│   │   ├── run_core_yoga_demo.py      # Demo end-to-end
│   │   ├── plot_pca_face.py           # PCA visualization
│   │   └── visualize_result.py        # So sánh skeleton
│   └── outputs/                       # (gitignore) model, case1, case2
│
└── YogaSinglePose-PTIT-main/          # Pipeline gốc PTIT
    ├── app.py                         # Demo webcam realtime
    ├── pose_estimation/               # MoveNet, MMPose, BlazePose
    ├── ml_classifier/                 # Train SVM, …
    ├── dataset/                       # (gitignore, ~1GB)
    ├── saved_models/                  # (gitignore)
    └── feature_ske_csv/               # (gitignore)
```

---

## 5. Thuật toán FACE

### 5.1 Ý tưởng

- **Nút đồ thị** = các mẫu trong tập **train** (mỗi nút gắn `image_path` thật).
- **Cạnh** = k láng giềng gần nhất (kNN).
- **Trọng số cạnh** (u, v):

  ```
  weight(u,v) = dist(u,v) / (density_avg(u,v) + ε)
  ```

  Đi qua vùng pose **thưa** (mật độ thấp theo KDE) bị **phạt nặng** → đường đi ưu tiên vùng “tự nhiên” của dữ liệu.

- **Nút query** = pose người dùng (ảnh sai hoặc confidence thấp), nối tới k neighbor gần nhất trên train.
- **Candidate** = mẫu train có `label == target` và model dự đoán target với `P ≥ threshold`.
- **Shortest path** (Dijkstra) từ query → candidate → lấy mẫu ở cuối path.
- **Fallback**: nếu không có path → chọn candidate gần nhất theo L2.

### 5.2 Triển khai trong code

| File | Nội dung |
|------|----------|
| `core_yoga_face/src/face_graph.py` | `build_face_graph()`, `find_counterfactual()` — NetworkX + sklearn KDE |
| `run_yoga_label_target_tests.py` | `build_face_adjacency()`, `shortest_path_to_candidates()` — Dijkstra thủ công + KDE |

### 5.3 Chuẩn hóa skeleton (`core_yoga_face`)

1. **Center**: dịch mid-hip về (0, 0).
2. **Scale**:
   - `torso`: chia cho khoảng cách mid-hip → mid-shoulder (mặc định).
   - `bbox`: chia cho đường chéo bounding box skeleton.
3. **Confidence**: giữ nguyên (hoặc bỏ với `--no_confidence`).

Pipeline root (`poses_yoga_multiclass.csv`) dùng feature **thô** `f0..f50`, **không** StandardScaler.

---

## 6. Hai case sử dụng

| Case | Điều kiện | Target label | Mục đích |
|------|-----------|--------------|----------|
| **Case 1** | Đúng nhãn nhưng `confidence < threshold` (mặc định 0.90) | **Cùng nhãn** | Sửa **tư thế** trong cùng động tác |
| **Case 2** | Dự đoán sai hoặc cần chuyển động tác | Nhãn đích (vd. `plank`, nhãn 5 trong thí nghiệm cũ) | Sửa **động tác** (action correction) |

Thí nghiệm root (`run_yoga_label_target_tests.py`) map thêm:

- **Hình 1** (`same_label_*.csv`): nhãn 1–4, confidence đúng nhãn &lt; 90% → gợi ý ảnh chuẩn cùng nhãn.
- **Hình 2** (`source_to_target_*.csv`): nhãn nguồn 1–4 → gợi ý sang **target label** (vd. 5).

---

## 7. Định dạng dữ liệu

### 7.1 CSV skeleton (`core_yoga_face`)

Hỗ trợ **hai cách đặt tên** 51 cột keypoint (code tự nhận diện khi load):

| Quy ước | Ví dụ cột | Ghi chú |
|---------|-----------|---------|
| Chỉ số `kp` | `kp0_x`, `kp0_y`, `kp0_c` … `kp16_*` | `kp0` = mũi, `kp9` = cổ tay trái, … |
| Tên MoveNet/COCO | `nose_x`, `left_wrist_x`, `left_wrist_score`, … | **Khuyến nghị** — dễ đọc, portable |

| Cột khác | Mô tả |
|----------|--------|
| `image_path` | Đường dẫn ảnh — **nên dùng tương đối** từ thư mục gốc repo FACE (xem [§13](#13-dataset-movenet_with_pathscsv-và-chuẩn-hóa-csv)) |
| `label` | Nhãn lớp (tên thư mục pose, vd. `bridge`, `warrior2`) |
| `pose_name`, `correctness`, `split` | (tuỳ chọn) |

**Ánh xạ nhanh (kp → tên):** kp0 mũi, kp5–6 vai, kp7–8 khuỷu, kp9–10 cổ tay, kp11–12 hông, kp13–14 gối, kp15–16 mắt cá.

### 7.2 CSV multiclass root (`poses_yoga_multiclass.csv`)

| Cột | Mô tả |
|-----|--------|
| `image_path` | Đường dẫn ảnh (tương đối dataset) |
| `f0` … `f50` | 51 feature đã trích sẵn |
| `label_name`, `label_id` | Tên và id lớp (vd. `downdog`, `warrior2`) |

### 7.3 File mẫu `core_yoga_face/data/movenet_with_paths.csv`

Bảng skeleton **đầy đủ** (~13 928 dòng) cho YogaSinglePose:

- Cột keypoint: `nose_x`, `left_wrist_x`, … (51 cột + `_score` thay `_c`)
- `image_path`: `YogaSinglePose-PTIT-main/dataset/<label>/<file>.jpg` (tương đối repo, không dùng `/Users/...`)
- `label`, `correctness`

Tách train/test:

```bash
cd core_yoga_face
python src/prepare_data.py \
  --input_csv data/movenet_with_paths.csv \
  --output_dir data/ \
  --label_col label
```

---

## 8. Cách cài đặt

### 8.1 Môi trường Python (khuyến nghị)

```bash
cd /path/to/FACE
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 8.2 Pipeline `core_yoga_face`

```bash
cd core_yoga_face
pip install -r requirements.txt
```

> **Lưu ý:** PyTorch là tuỳ chọn. Chỉ cần `mlp` (sklearn) hoặc `rf` thì không bắt buộc cài torch.

### 8.3 YogaSinglePose (trích skeleton / webcam)

Cần thêm dependency theo `YogaSinglePose-PTIT-main` (OpenCV, TensorFlow/TFLite hoặc MMPose tuỳ script). Thư mục `dataset/` và `saved_models/` phải có trên máy local (không có trên GitHub).

---

## 9. Hướng dẫn chạy từng pipeline

### 9.1 Pipeline A — `core_yoga_face` (khuyến nghị)

Đây là luồng **đầy đủ** train → FACE demo → visualization.

#### Bước 0: Chuẩn bị dữ liệu (nếu chưa có CSV)

```bash
cd core_yoga_face

# Sinh dữ liệu mẫu (demo)
python src/prepare_data.py \
  --generate_sample \
  --n_samples 500 \
  --n_classes 5 \
  --output_dir data/

# Hoặc tách từ CSV skeleton đã có (khuyến nghị)
python src/prepare_data.py \
  --input_csv data/movenet_with_paths.csv \
  --output_dir data/ \
  --test_size 0.2 \
  --label_col label

# Hoặc tách từ CSV raw khác
python src/prepare_data.py \
  --input_csv data/raw_skeleton.csv \
  --output_dir data/ \
  --test_size 0.2 \
  --label_col label
```

#### Bước 1: Train classifier

```bash
# MLP sklearn (khuyến nghị)
python src/train_classifier.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_type mlp \
  --output_dir outputs/model

# Random Forest
python src/train_classifier.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_type rf \
  --output_dir outputs/model
```

**Artifact trong `outputs/model/`:** `model.pkl`, `scaler.pkl`, `label_encoder.pkl`, `metrics.json`, `config.json`.

#### Bước 2a: Case 1 — Sửa tư thế (cùng nhãn)

```bash
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_dir outputs/model \
  --case 1 \
  --threshold 0.90 \
  --output_dir outputs/case1
```

#### Bước 2b: Case 2 — Chuyển sang động tác đích

```bash
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_dir outputs/model \
  --case 2 \
  --target_label plank \
  --threshold 0.90 \
  --output_dir outputs/case2
```

#### Chọn mẫu cụ thể

```bash
# Theo index trong test CSV
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_dir outputs/model \
  --case 1 \
  --input_index 42 \
  --output_dir outputs/case1

# Theo image_path
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --model_dir outputs/model \
  --case 2 \
  --input_image_path "path/to/image.jpg" \
  --output_dir outputs/case2
```

#### Tham số FACE graph (tuỳ chỉnh)

| Tham số | Mặc định | Mô tả |
|---------|----------|--------|
| `--k_graph` | 7 | Số láng giềng kNN |
| `--k_density` | 5 | k cho ước lượng mật độ |
| `--top_k_candidates` | 50 | Số candidate tối đa |
| `--threshold` | 0.90 | Ngưỡng confidence |

---

### 9.2 Pipeline B — Root label-target test

Dùng `poses_yoga_multiclass.csv`, **ExtraTrees**, feature **không chuẩn hóa**, FACE graph với KDE.

```bash
cd /path/to/FACE

python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_test_plan \
  --dataset-root .
```

**Chạy mọi nhãn làm target:**

```bash
python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_all_targets \
  --dataset-root . \
  --all-targets
```

**Tuỳ chỉnh FACE graph:**

```bash
python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_test_plan \
  --dataset-root . \
  --face-k-neighbors 20 \
  --kde-bandwidth 0.25 \
  --density-weight 0.35
```

**Bỏ qua bước render ảnh:**

```bash
python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_test_plan \
  --dataset-root . \
  --skip-render
```

> `--dataset-root` dùng để resolve đường dẫn `image_path` tương đối khi render cặp ảnh. Đặt trỏ tới thư mục chứa `yoga_poses_dataset/` nếu ảnh nằm trong dataset gốc.

---

### 9.3 Pipeline C — Render cặp ảnh từ CSV

Sau khi chạy `run_yoga_label_target_tests.py`:

```bash
python render_counterfactual_image_pairs.py \
  --counterfactual-csv outputs_label5_test_plan/same_label_1-1_to_4-4.csv \
  --output-dir outputs_label5_test_plan/ht1_same_label_images \
  --dataset-root .
```

---

### 9.4 Pipeline D — YogaSinglePose realtime (webcam)

```bash
cd YogaSinglePose-PTIT-main
# Cần saved_models/SVM.joblib và model MoveNet TFLite
python app.py
```

Luồng: webcam → MoveNet extract → SVM predict → hiển thị nhãn (tiếng Việt). **Không** chạy FACE graph trong `app.py`.

---

### 9.5 Trích skeleton từ ảnh (offline)

Trong `YogaSinglePose-PTIT-main/pose_estimation/`:

- `blazepose_movenet_run_output_csv.py` — MoveNet / BlazePose → CSV
- `mmpose_run_output_csv.py` — MMPose → CSV

Xem `YogaSinglePose-PTIT-main/pose_estimation/README.md` để biết lệnh chi tiết và đường dẫn model.

---

## 10. Kết quả đầu ra

### 10.1 `core_yoga_face/outputs/case1/` (ví dụ)

| File | Nội dung |
|------|----------|
| `result.json` | Kết quả đầy đủ FACE |
| `pca_case1.png` | PCA + đường đi FACE |
| `result_visualization.png` | So sánh skeleton (nếu bật) |
| `mlp_prediction_*.json` | Chi tiết dự đoán MLP |

**Ví dụ trường trong `result.json`:**

```json
{
  "input_image_path": "...",
  "ground_truth_label": "warrior2",
  "predicted_label": "warrior2",
  "predicted_conf": 0.72,
  "target_label": "warrior2",
  "counterfactual_path": "data/images/correct_001.jpg",
  "face_path": [200, 45, 12, 89],
  "face_path_length": 4,
  "face_path_cost": 2.31,
  "method": "FACE shortest path"
}
```

### 10.2 `outputs_label5_test_plan/` (root experiment)

| File | Mô tả |
|------|--------|
| `same_label_1-1_to_4-4.csv` | Case cùng nhãn (hình 1) |
| `source_to_target_1-5_to_4-5.csv` | Case chuyển nhãn (hình 2) |
| `pca_raw_no_normalize.png` | PCA feature thô + đường FACE |
| `ht1_same_label_images/counterfactual_image_pairs.png` | Cặp ảnh hình 1 |
| `ht2_source_to_target_images/counterfactual_image_pairs.png` | Cặp ảnh hình 2 |
| `metrics_no_normalize.csv` | Accuracy, F1, … |

---

## 11. Lưu ý Git và dữ liệu lớn

File `.gitignore` loại trừ:

- `YogaSinglePose-PTIT-main/dataset/` (~1.1 GB)
- `saved_models/`, `feature_ske_csv/`, `saved_logs/`
- `core_yoga_face/outputs/`, `outputs/`, `outputs_*/`
- `*.pth`, `*.pt`, `*.pkl`, `*.mp4`, `.venv/`

Sau khi clone repo, cần **tự chuẩn bị** dataset ảnh và model train sẵn để chạy đầy đủ pipeline trích xuất và demo webcam.

**Git trên macOS:** Nếu lệnh `git commit` báo lỗi `unknown option trailer`, dùng `/usr/bin/git` thay vì bản Git cũ trong `/usr/local/bin/git`.

---

## 12. Xử lý sự cố

| Triệu chứng | Gợi ý |
|-------------|--------|
| `Missing required columns: 51 keypoint columns` | Kiểm tra tên cột `kp{i}_x/y/c` hoặc `nose_x`, `left_eye_x`, … |
| `No candidates with label=X and prob>=0.90` | Giảm `--threshold` (0.8, 0.7) hoặc mở rộng pool train |
| FACE path quá dài / không hợp lý | Tăng `--k_graph` (10–15) hoặc `--face-k-neighbors` |
| Render ảnh lỗi “file not found” | Clone đủ `YogaSinglePose-PTIT-main/dataset/`; `image_path` phải là đường dẫn tương đối như trong CSV; chạy lệnh từ thư mục gốc FACE |
| Không có `outputs/model/` | Chạy `train_classifier.py` trước `run_core_yoga_demo.py` |
| KDE / graph lỗi | Pipeline tự fallback nearest L2; xem `method` trong JSON |

---

## 13. Dataset `movenet_with_paths.csv` và chuẩn hóa CSV

### 13.1 Chuẩn mới (2026-06)

| Trước | Sau |
|-------|-----|
| `kp9_x`, `kp9_y`, `kp9_c` | `left_wrist_x`, `left_wrist_y`, `left_wrist_score` |
| `/Users/.../FACE/YogaSinglePose-PTIT-main/dataset/...` | `YogaSinglePose-PTIT-main/dataset/...` |

`load_dataset.resolve_image_path()` tự tìm ảnh từ thư mục gốc repo khi train/demo/visualize.

### 13.2 Script chuẩn hóa CSV khác

```bash
cd core_yoga_face
python src/normalize_skeleton_csv.py \
  --input data/raw_skeleton.csv \
  --output data/raw_skeleton.csv \
  --inplace
```

Tham số `--repo-root` (tuỳ chọn): chỉ định thư mục gốc FACE nếu không nằm cạnh `core_yoga_face/`.

### 13.3 Sau khi clone repo (máy mới)

1. `git clone https://github.com/ITIS-mphuong169/FACE.git && cd FACE`
2. Có file `core_yoga_face/data/movenet_with_paths.csv` trên Git
3. Tải/copy ảnh vào `YogaSinglePose-PTIT-main/dataset/<label>/` **đúng cấu trúc** như trong cột `image_path`
4. `cd core_yoga_face && pip install -r requirements.txt`
5. `python src/prepare_data.py --input_csv data/movenet_with_paths.csv --output_dir data/`

---

## 14. Phụ lục — Tạo lại toàn bộ CSV từ đầu

Nếu đã xóa `train.csv`, `test.csv`, `movenet_with_paths.csv`:

### A — Thử nhanh (không cần ảnh)

```bash
cd core_yoga_face
python src/prepare_data.py --generate_sample --n_samples 500 --n_classes 5 --output_dir data/
```

### B — Từ ảnh thật (YogaSinglePose)

**0. Cấu trúc ảnh**

```
YogaSinglePose-PTIT-main/dataset/
├── bridge/
│   └── *.jpg
├── tree/
└── ...
```

**1. Trích skeleton**

```bash
cd YogaSinglePose-PTIT-main
pip install opencv-python pandas numpy tensorflow tensorflow-hub
python pose_estimation/blazepose_movenet_run_output_csv.py \
  --dataset_dir dataset \
  --output_dir feature_ske_csv \
  --model_name movenet_tflite
```

**2. Gắn `image_path` + đổi `pose_class` → `label`**

Chạy script Python (cùng thứ tự ảnh với bước 1) hoặc dùng `normalize_skeleton_csv.py` sau khi merge thủ công thành CSV có cột `kp*`.

**3. Chuẩn hóa tên cột + đường dẫn**

```bash
cd core_yoga_face
python src/normalize_skeleton_csv.py --input data/raw_skeleton.csv --inplace
```

**4–6.** `prepare_data.py` → `train_classifier.py` → `run_core_yoga_demo.py` (xem [§9.1](#91-pipeline-a--core_yoga_face-khuyến-nghị)).

---

## Phụ lục — Luồng một lần chạy (checklist)

**Muốn demo FACE nhanh nhất (có dataset skeleton trên Git):**

1. `cd core_yoga_face && pip install -r requirements.txt`
2. `python src/prepare_data.py --input_csv data/movenet_with_paths.csv --output_dir data/` (hoặc đã có `train.csv` / `test.csv`)
3. Có thư mục ảnh `../YogaSinglePose-PTIT-main/dataset/` nếu cần xem ảnh gợi ý
4. `python src/train_classifier.py --train_csv data/train.csv --test_csv data/test.csv --model_type mlp --output_dir outputs/model`
5. `python src/run_core_yoga_demo.py --train_csv data/train.csv --test_csv data/test.csv --model_dir outputs/model --case 1 --output_dir outputs/case1`
6. Mở `outputs/case1/result.json` và `pca_case1.png`

**Muốn chạy thí nghiệm multiclass trên CSV có sẵn ở root:**

1. `pip install -r requirements.txt` (từ thư mục FACE)
2. `python run_yoga_label_target_tests.py --input-csv poses_yoga_multiclass.csv --output-dir outputs_label5_test_plan --dataset-root .`
3. Xem CSV và `pca_raw_no_normalize.png` trong thư mục output

---

