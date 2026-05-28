import cv2
import numpy as np
from typing import List, Tuple

from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
register_all_modules()

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]
POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)

COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

class COCOPoseExtractor:
    def __init__(self, config_file, checkpoint_file, device='cpu'):
        self.model = init_model(config_file, checkpoint_file, device=device)
        self.keypoint_names = COCO_KEYPOINT_NAMES

    def _normalize(self, keypoints):
        kp = keypoints.copy()
        left_hip = kp[11][:2]
        right_hip = kp[12][:2]
        hip_center = (left_hip + right_hip) / 2
        left_shoulder = kp[5][:2]
        right_shoulder = kp[6][:2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        max_dist = np.max(np.linalg.norm(kp[:, :2] - hip_center, axis=1))
        scale = max(torso_size * 2.5, max_dist)
        norm_kp = []
        for x, y, score in kp:
            nx = (x - hip_center[0]) / scale
            ny = (y - hip_center[1]) / scale
            norm_kp.extend([nx, ny, score])
        return norm_kp

    def draw_pose(self, image, keypoints):
        if image is None:
            return None
        img = image.copy()
        for idx, (x, y) in enumerate(keypoints[:, :2]):
            if keypoints[idx][2] > 0.3:
                cv2.circle(img, (int(x), int(y)), 4, POINT_COLOR, -1)
        for i, j in COCO_SKELETON:
            if keypoints[i][2] > 0.3 and keypoints[j][2] > 0.3:
                pt1 = tuple(keypoints[i][:2].astype(int))
                pt2 = tuple(keypoints[j][:2].astype(int))
                cv2.line(img, pt1, pt2, LINE_COLOR, 2)
        return img

    def extract_from_image(self, image: np.ndarray, show=False) -> Tuple[List[float], bool, np.ndarray]:
        result = inference_topdown(self.model, image)
        if (len(result) == 0 or 
            not hasattr(result[0].pred_instances, 'keypoints') or 
            len(result[0].pred_instances.keypoints) == 0 or 
            not hasattr(result[0].pred_instances, 'keypoint_scores') or 
            len(result[0].pred_instances.keypoint_scores) == 0 or
            np.mean(result[0].pred_instances.keypoint_scores[0]) < 0.3):
            return np.zeros(17 * 3), False, None
        keypoints = result[0].pred_instances.keypoints[0]
        scores = result[0].pred_instances.keypoint_scores[0]
        keypoints_3d = np.hstack([keypoints, scores[:, np.newaxis]])
        normalized = self._normalize(keypoints_3d)
        vis_img = self.draw_pose(image, keypoints_3d) if show else None
        return normalized, True, vis_img

    def create_column_names(self):
        columns = []
        for name in self.keypoint_names:
            for attr in ['x', 'y', 'score']:
                columns.append(f'{name}_{attr}')
        return columns

# Test code for COCOPoseExtractor can be added here if needed

if __name__ == "__main__":
    # Đường dẫn tới config và checkpoint của mô hình MMPose
    CONFIG = 'pose_estimation/model_ske/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
    CHECKPOINT = 'pose_estimation/model_ske/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

    # Đường dẫn tới ảnh đầu vào
    IMAGE_PATH = 'dataset/bridge/br1_0001.jpg'
    # IMAGE_PATH = 'dataset\\child\\ch2_0013.jpg'
    IMAGE_PATH = "dataset\\child\\ch2_0013.jpg"
    # Khởi tạo extractor
    extractor = COCOPoseExtractor(CONFIG, CHECKPOINT, device='cpu')  # hoặc 'cuda' nếu bạn có GPU
    image_np = cv2.imread(IMAGE_PATH)  # hoặc ảnh từ webcam...
    # Chạy extract + visualize
    keypoints_vector, success, vis_img = extractor.extract_from_image(image_np, show=True)
    print(keypoints_vector, success, vis_img)
    if success and vis_img is not None:
        cv2.imshow("Pose Estimation", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không phát hiện được người trong ảnh.")
