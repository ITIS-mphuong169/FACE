# CORE-Yoga FACE

**Feasible and Actionable Counterfactual Explanations cho bài toán phân loại yoga skeleton**

---

## Tổng quan

CORE-Yoga FACE là pipeline Python giúp:
1. **Phân loại tư thế yoga** từ skeleton 17 keypoints (MoveNet) → 51 features.
2. **Tìm counterfactual khả thi** bằng thuật toán FACE graph khi model dự đoán sai hoặc confidence thấp.
3. **Trực quan hóa** kết quả bằng PCA scatter plot và so sánh skeleton.

### Hai case chính

| Case | Điều kiện kích hoạt | Target label |
|------|---------------------|--------------|
| **Case 1** | Đúng nhãn nhưng confidence < 0.90 | Cùng nhãn (posture correction) |
| **Case 2** | Dự đoán sai label | Label 5 (hoặc chỉ định) |

---

## Cấu trúc project

```
core_yoga_face/
├── data/
│   ├── train.csv          # skeleton CSV train
│   └── test.csv           # skeleton CSV test
├── outputs/
│   ├── model/             # model artifacts
│   ├── case1/             # kết quả case 1
│   └── case2/             # kết quả case 2
├── src/
│   ├── load_dataset.py        # load & normalize skeleton
│   ├── train_classifier.py    # train MLP / RandomForest
│   ├── face_graph.py          # FACE graph + counterfactual
│   ├── run_core_yoga_demo.py  # main demo pipeline
│   ├── plot_pca_face.py       # PCA visualization
│   ├── visualize_skeleton.py  # skeleton comparison
│   └── prepare_data.py        # data preparation utility
├── requirements.txt
└── README.md
```

---

## Cài đặt

```bash
cd core_yoga_face
pip install -r requirements.txt
```

> **Lưu ý**: PyTorch là optional. Nếu chỉ dùng `mlp` (sklearn) hoặc `rf`, không cần cài torch.

---

## Định dạng CSV

CSV phải có tối thiểu các cột sau:

| Cột | Mô tả |
|-----|-------|
| `image_path` | Đường dẫn đến ảnh gốc |
| `label` | Nhãn lớp (int hoặc str) |
| `kp0_x`, `kp0_y`, `kp0_c` | Keypoint 0: x, y, confidence |
| ... | (tiếp tục đến kp16) |
| `pose_name` | (optional) tên pose |
| `correctness` | (optional) `correct` / `incorrect` |
| `split` | (optional) `train` / `test` |

**Tổng cộng 51 cột keypoint**: `kp{i}_x`, `kp{i}_y`, `kp{i}_c` với `i` từ 0 đến 16.

---

## Cách chạy

### 0. Chuẩn bị dữ liệu

Nếu chưa có CSV, dùng script sinh dữ liệu mẫu:

```bash
python src/prepare_data.py \
  --generate_sample \
  --n_samples 500 \
  --n_classes 5 \
  --output_dir data/
```

Nếu đã có CSV raw:

```bash
python src/prepare_data.py \
  --input_csv data/raw_skeleton.csv \
  --output_dir data/ \
  --test_size 0.2 \
  --label_col label
```

---

### 1. Train classifier

```bash
# MLP (sklearn) — recommended
python src/train_classifier.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_type mlp \
  --output_dir outputs/model

# Random Forest
python src/train_classifier.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_type rf \
  --output_dir outputs/model

# PyTorch MLP (cần cài torch)
python src/train_classifier.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_type mlp_torch \
  --output_dir outputs/model
```

**Output:**
```
outputs/model/
  model.pkl        # trained classifier
  scaler.pkl       # StandardScaler
  label_encoder.pkl
  metrics.json     # accuracy, confusion matrix
  config.json      # preprocessing config
```

---

### 2. Chạy FACE demo

#### Case 1 — Posture correction (cùng nhãn)

```bash
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_dir outputs/model \
  --case 1 \
  --threshold 0.90 \
  --output_dir outputs/case1
```

#### Case 2 — Action correction (sang nhãn 5)

```bash
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_dir outputs/model \
  --case 2 \
  --target_label 5 \
  --threshold 0.90 \
  --output_dir outputs/case2
```

#### Chọn sample cụ thể

```bash
# Theo index trong test CSV
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_dir outputs/model \
  --case 1 \
  --input_index 42 \
  --output_dir outputs/case1

# Theo image_path
python src/run_core_yoga_demo.py \
  --train_csv data/train.csv \
  --test_csv  data/test.csv \
  --model_dir outputs/model \
  --case 2 \
  --input_image_path "data/images/warrior_001.jpg" \
  --output_dir outputs/case2
```

---

### 3. Output của demo

```
outputs/case1/
  result.json          # kết quả đầy đủ
  pca_case1.png        # PCA FACE visualization
  skeleton_compare.png # so sánh skeleton
```

Nội dung `result.json`:
```json
{
  "input_image_path": "...",
  "ground_truth_label": "1",
  "predicted_label": "1",
  "predicted_conf": 0.72,
  "gt_conf": 0.72,
  "target_label": "1",
  "counterfactual_path": "data/images/correct_001.jpg",
  "counterfactual_label": "1",
  "counterfactual_prob": 0.95,
  "face_path": [200, 45, 12, 89],
  "face_path_length": 4,
  "face_path_cost": 2.31,
  "method": "FACE shortest path"
}
```

---

## Tham số dòng lệnh

### `train_classifier.py`

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--train_csv` | required | CSV train |
| `--test_csv` | required | CSV test |
| `--model_type` | `mlp` | `mlp`, `rf`, `mlp_torch` |
| `--output_dir` | `outputs/model` | Thư mục lưu model |
| `--label_col` | `label` | Tên cột nhãn |
| `--no_confidence` | false | Bỏ kênh confidence |
| `--scale_mode` | `torso` | `torso`, `bbox`, `none` |
| `--no_center` | false | Không center skeleton |

### `run_core_yoga_demo.py`

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `--train_csv` | required | |
| `--test_csv` | required | |
| `--model_dir` | required | Thư mục chứa model artifacts |
| `--case` | required | `1` hoặc `2` |
| `--threshold` | `0.90` | Ngưỡng confidence |
| `--input_index` | auto | Index trong test CSV |
| `--input_image_path` | None | image_path cụ thể |
| `--target_label` | None | Override target (Case 2 default=5) |
| `--k_graph` | `7` | k cho kNN graph |
| `--k_density` | `5` | k cho density estimation |
| `--top_k_candidates` | `50` | Số candidates tối đa |
| `--skip_pca` | false | Bỏ qua PCA plot |
| `--skip_skeleton` | false | Bỏ qua skeleton plot |
| `--run_all_test` | false | Chạy toàn bộ test set |

---

## Thuật toán FACE

```
1. Build kNN graph trên train set
   - Edge weight = dist(u,v) / (density_avg(u,v) + ε)
   - Density ≈ KernelDensity (fallback: 1/mean_kNN_dist)

2. Add query node (x_input) → connect to k nearest train nodes

3. Find shortest path (Dijkstra) từ query → candidate nodes
   - Candidate: label == target AND P(target) >= threshold

4. Return train sample tại end of path → image_path guaranteed real

5. Fallback (nếu graph lỗi hoặc không có path):
   - Chọn candidate gần nhất theo Euclidean distance
   - Log: method = "fallback nearest feasible"
```

---

## Normalization skeleton

```
1. Center: dịch mid-hip về gốc tọa độ (0,0)
2. Scale:
   - "torso": chia cho khoảng cách mid-hip → mid-shoulder
   - "bbox": chia cho đường chéo bounding box skeleton
3. Confidence: giữ nguyên (hoặc bỏ với --no_confidence)
```

---

## Troubleshooting

**Lỗi: "Missing required columns: 51 keypoint columns"**
→ Kiểm tra tên cột CSV có đúng định dạng `kp{i}_x`, `kp{i}_y`, `kp{i}_c` không.

**Lỗi: "No candidates with label=X and prob>=0.90"**
→ Giảm `--threshold` xuống 0.80 hoặc 0.70.

**PCA plot không có confidence background**
→ Model không hỗ trợ `predict_proba`. Dùng `mlp` thay vì `rf` với `max_features` khác.

**FACE path rất dài hoặc không hợp lý**
→ Tăng `--k_graph` (thử 10-15) hoặc tăng `--top_k_candidates`.

---

## Dataset: YogaSinglePose-PTITI

Dataset gốc sử dụng MoveNet để trích xuất skeleton từ ảnh yoga.
Sau khi extract, CSV có cấu trúc:

```
image_path, label, kp0_x, kp0_y, kp0_c, ..., kp16_x, kp16_y, kp16_c
```

Sử dụng script `prepare_data.py` để tách train/test split.

---

## License

MIT License
