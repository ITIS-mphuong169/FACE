import cv2
import numpy as np
import pandas as pd
import os
import glob
import time
import argparse
from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
from mmpose_coco_extractor import COCOPoseExtractor
from pathlib import Path

register_all_modules()
<<<<<<< HEAD

models = [
    # {
    #     'name': 'hrnet-w48',
    #     'config': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
    #     'checkpoint': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    # },
    {
        'name': 'hrnet-w32',
        'config': 'td-hm_hrnet-w32_fp16-8xb64-210e_coco-256x192.py',
        'checkpoint': 'td-hm_hrnet-w32_fp16-8xb64-210e_coco-256x192-f1e84e3b_20220914.pth'
    },
    # {
    #     'name': 'rtmpose-s',
    #     'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
    #     'checkpoint': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
    # },
        {
        'name': 'rtmpose-t',
        'config': 'rtmpose-t_8xb256-420e_aic-coco-256x192.py',
        'checkpoint': 'rtmpose-tiny_simcc-aic-coco_pt-aic-coco_420e-256x192-cfc8f33d_20230126.pth'
    },
        
=======
models = [
    {
        'name': 'hrnet-w48',
        'config': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py',
        'checkpoint': 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
    },
    {
        'name': 'rtmpose-s',
        'config': 'rtmpose-s_8xb256-420e_coco-256x192.py',
        'checkpoint': 'rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
    },
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    {
        'name': 'rtmo-s',
        'config': 'rtmo-s_8xb32-600e_coco-640x640.py',
        'checkpoint': 'rtmo-s_8xb32-600e_coco-640x640-8db55a59_20231211.pth'
    },
<<<<<<< HEAD

    # {
    #     'name': 'yoloxpose-s',
    #     'config': 'yoloxpose_s_8xb32-300e_coco-640.py',
    #     'checkpoint': 'yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth'
    # },
      {
        'name': 'yoloxpose-t',
        'config': 'yoloxpose_tiny_4xb64-300e_coco-416.py',
        'checkpoint': 'yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth'
    },
    # {
    #     'name': 'dekr_hrnet-w32',
    #     'config': 'dekr_hrnet-w32_8xb10-140e_coco-512x512.py',
    #     'checkpoint': 'dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth'
    # },
    # {
    #     'name': 'td-reg_res50',
    #     'config': 'td-reg_res50_8xb64-210e_coco-256x192.py',
    #     'checkpoint': 'td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.pth'
    # }
          {
        'name': 'vitpose-s',
        'config': 'td-hm_ViTPose-small_8xb64-210e_coco-256x192.py',
        'checkpoint': 'td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth'
    },

]



=======
    {
        'name': 'yoloxpose-s',
        'config': 'yoloxpose_s_8xb32-300e_coco-640.py',
        'checkpoint': 'yoloxpose_s_8xb32-300e_coco-640-56c79c1f_20230829.pth'
    },
    {
        'name': 'dekr_hrnet-w32',
        'config': 'dekr_hrnet-w32_8xb10-140e_coco-512x512.py',
        'checkpoint': 'dekr_hrnet-w32_8xb10-140e_coco-512x512_ac7c17bf-20221228.pth'
    },
    {
        'name': 'td-reg_res50',
        'config': 'td-reg_res50_8xb64-210e_coco-256x192.py',
        'checkpoint': 'td-reg_res50_8xb64-210e_coco-256x192-72ef04f3_20220913.pth'
    }
]
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
def main(args):

    # Chỉ chạy một mô hình theo tên truyền vào
    model = next((m for m in models if m['name'] == args.model_name), None)
    if not model:
        print(f"❌ Model name '{args.model_name}' not found.")
        return

    model['config'] = os.path.join("pose_estimation/model_ske/", model['config'])
    model['checkpoint'] = os.path.join("pose_estimation/model_ske/", model['checkpoint'])

    print(f"Config: {model['config']}")
    print(f"Checkpoint: {model['checkpoint']}")
    if not os.path.exists(model['config']) or not os.path.exists(model['checkpoint']):
        print(f"❌ Model files not found, skipping...")
        return

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
        return

    print(f"Found {len(image_paths)} images to process...")

<<<<<<< HEAD
    
    extractor = COCOPoseExtractor(model['config'], model['checkpoint'], "cpu")
    start_time = time.time()
=======
    start_time = time.time()
    extractor = COCOPoseExtractor(model['config'], model['checkpoint'], "cpu")

>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
    all_data = []
    all_labels = []
    all_hpe_status =[]
    for i, image_path in enumerate(image_paths, 1):
        folder_name = os.path.basename(os.path.dirname(image_path))
        pose_class_label = folder_name
        landmarks, success, vis_img = extractor.extract_from_image(image_path, show=False)

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

    elapsed = time.time() - start_time
    print(f"⏱️ Model '{model['name']}' finished in {elapsed:.2f} seconds.")
<<<<<<< HEAD
    print(f"⏳ Average time per image: {(elapsed / len(image_paths)) * 1000:.2f} ms")
=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644

    if all_data:
        model_name = os.path.splitext(os.path.basename(model['config']))[0]
        os.makedirs(args.output_dir, exist_ok=True)
        output_csv = os.path.join(args.output_dir, f'pose_data_{model_name}.csv')

        df = pd.DataFrame(all_data, columns=extractor.create_column_names())
        df['pose_class'] = all_labels
        df['hpe_status'] = all_hpe_status
        df.to_csv(output_csv, index=False)
        print(f"✅ All data saved to {output_csv}")
        print(f"📊 Total processed images: {len(all_data)}")
    else:
        print("⚠️ No valid keypoints extracted from any image.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose estimation for a specific model and save results.")
    parser.add_argument("--dataset_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="feature_ske_csv", help="Path to output directory for CSV")
    parser.add_argument("--model_name", type=str, default="hrnet-w48", help="Name of the model to run (e.g. hrnet-w48, rtmpose-s, rtmo-s, yoloxpose-s, dekr_hrnet-w32, td-reg_res50)")
    args = parser.parse_args()
    main(args)
