import cv2
import numpy as np
import pandas as pd
import os
import glob
import time
import argparse
from blazepose_movenet_extractors import BlazePoseExtractor, BlazePoseCOCOExtractor,MoveNetPoseExtractor,MoveNetPoseTFLiteExtractor

from pathlib import Path

# Danh sách các model bạn muốn chạy, mỗi phần tử là dict có tên class (class đã import)
models = [
    {"name": "blazepose_33", "class": BlazePoseExtractor},
    {"name": "blazepose_coco_17", "class": BlazePoseCOCOExtractor},
    {"name": "movenet", "class": MoveNetPoseExtractor},
    {"name": "movenet_tflite", "class": MoveNetPoseTFLiteExtractor},
]

def main(args):
    
    # Lấy tất cả file ảnh jpg/jpeg/png trong thư mục chỉ định
    image_paths = [
        Path(p).as_posix()
        for p in (
            glob.glob(f'{args.dataset_dir}/**/*.jpg', recursive=True) +
            glob.glob(f'{args.dataset_dir}/**/*.jpeg', recursive=True) +
            glob.glob(f'{args.dataset_dir}/**/*.png', recursive=True)
        )
    ]

<<<<<<< HEAD
    # image_paths=image_paths[0:10]
=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    if not image_paths:
        print("❌ No image files found.")
        exit()

    print(f"Found {len(image_paths)} images to process...")

    os.makedirs(args.output_dir, exist_ok=True)

    # Chọn model theo tên nếu truyền vào, nếu không thì chạy tất cả
    if args.model_name:
        selected_models = [m for m in models if m["name"] == args.model_name]
        if not selected_models:
            print(f"❌ Model '{args.model_name}' not found in available models: {[m['name'] for m in models]}")
            exit()
    else:
        selected_models = models

    # Chỉ chạy với một model (lấy model đầu tiên trong selected_models)
    model_info = selected_models[0]
    print(f"\n=== Processing Model: {model_info['name']} ===")
    start_time = time.time()
    extractor = model_info["class"]()
    all_data = []
    all_labels = []
    all_hpe_status = []
    for i, image_path in enumerate(image_paths, 1):
        # Lấy nhãn từ tên thư mục chứa file ảnh
        folder_name = os.path.basename(os.path.dirname(image_path))
        pose_class_label = folder_name
        image_np = cv2.imread(image_path)
        landmarks, success, vis_img = extractor.extract_from_image(image_np, show=False)

        if success and landmarks is not None:
            all_data.append(landmarks)
            all_labels.append(pose_class_label)
            all_hpe_status.append(1)
            # print(f"✅ Successfully processed: {image_path} (label: {pose_class_label})")
        else:
            all_data.append(landmarks)
            all_labels.append(pose_class_label)
            all_hpe_status.append(0)
            print(f"❌ Failed to process image: {image_path}")


    # Ghi dữ liệu keypoints ra file CSV theo từng model
    if all_data:
        output_csv = f'{args.output_dir}/pose_data_{model_info["name"]}.csv'
        df = pd.DataFrame(all_data, columns=extractor.create_column_names())
        df['pose_class'] = all_labels
        df['hpe_status'] = all_hpe_status
        df.to_csv(output_csv, index=False)
        print(f"✅ All data saved to {output_csv}")
        print(f"📊 Total processed images: {len(all_data)}")
    else:
        print(f"⚠️ No valid keypoints extracted from any image by model {model_info['name']}.")

    elapsed = time.time() - start_time
    print(f"⏱️ Model '{model_info['name']}' finished in {elapsed:.2f} seconds.")
<<<<<<< HEAD
    print(f"⏳ Average time per image: {(elapsed / len(image_paths)) * 1000:.2f} ms")
=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train all ML models and save results.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory.")
    parser.add_argument("--output_dir", type=str, default="feature_ske_csv", help="Directory to save output CSV files.")
    parser.add_argument("--model_name", type=str, default="movenet", help="Name of model to run (run all if not specified).")
    args = parser.parse_args()
    main(args)
