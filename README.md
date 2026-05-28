# FACE Pose Correction Pipeline

This project implements a prototype of the FACE algorithm for human pose correction based on:

- **Step 1 – Feature Extraction (MoveNet)**: extract 17 keypoints from an RGB image.
- **Step 2 – Classification (MLP/Transformer)**: decide whether a pose is correct or incorrect.
- **Step 3 – FACE Graph Search**: build a graph on correct poses, weight edges with a density-aware metric, and search for a plausible counterfactual pose.
- **Step 4 – Actionable Feedback**: convert pose differences into natural language instructions.

## Project structure

- `run_yoga_label_target_tests.py`: current label-target experiment pipeline.
- `render_counterfactual_image_pairs.py`: renders wrong/suggested image pairs from generated CSV outputs.
- `poses_yoga_multiclass.csv`: current multiclass pose feature dataset.
- `outputs_label5_test_plan/`: latest single-target experiment outputs.
- `outputs_label5_all_targets/`: all-target experiment outputs.
- `unused/`: archived scripts, old outputs, and datasets that are not part of the current root workflow.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Notes

- The current root workflow uses the prepared `poses_yoga_multiclass.csv` feature table.
- Older binary, MoveNet extraction, and multiclass helper pipelines are archived under `unused/legacy_root_scripts/`.

## Label-target test plan, no normalization

Chạy test từ dataset cũ `poses_yoga_multiclass.csv`, tự tách train/test, dùng ExtraTrees không scale feature và chọn ảnh gợi ý bằng FACE-like graph search. Đồ thị nối các pose lân cận bằng kNN, trọng số cạnh dùng khoảng cách L2 có phạt vùng mật độ thấp theo KDE, rồi chọn counterfactual có shortest path tốt nhất.

```bash
python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_test_plan \
  --dataset-root .
```

Output chính:

- `same_label_1-1_to_4-4.csv`: hình 1, động tác đúng nhãn nhưng confidence đúng nhãn `<90%`, sửa sang ảnh đúng cùng nhãn `1-1`, `2-2`, `3-3`, `4-4`.
- `source_to_target_1-5_to_4-5.csv`: hình 2, động tác từ nhãn `1..4` sửa sang nhãn target `5`.
- `ht1_same_label_images/counterfactual_image_pairs.png`: hình cặp ảnh cho hình 1, không vẽ skeleton.
- `ht2_source_to_target_images/counterfactual_image_pairs.png`: hình cặp ảnh cho hình 2, không vẽ skeleton.
- `pca_raw_no_normalize.png`: PCA trên feature thô, không StandardScaler/normalize, vẽ đường đi FACE graph cho hình 1 và hình 2.

Tham số FACE-like graph có thể chỉnh:

```bash
python run_yoga_label_target_tests.py \
  --input-csv poses_yoga_multiclass.csv \
  --output-dir outputs_label5_test_plan \
  --dataset-root . \
  --face-k-neighbors 20 \
  --kde-bandwidth 0.25 \
  --density-weight 0.35
```
