import cv2
import numpy as np
from typing import List, Tuple

# BlazePose
import mediapipe as mp

# MoveNet
import tensorflow as tf
import tensorflow_hub as hub

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12),
    (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3),
    (2, 4), (3, 5), (4, 6)
]
POINT_COLOR = (0, 0, 255)
LINE_COLOR = (0, 255, 0)

COCO_KEYPOINTS_IDX = [
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28
]
COCO_KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

class BlazePoseExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.3
        )
        self.landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky', 'right_pinky',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index'
        ]
        self.skeleton = self.mp_pose.POSE_CONNECTIONS  # Use BlazePose skeleton

    def _normalize(self, landmarks) -> List[float]:
        left_hip = landmarks[self.landmark_names.index('left_hip')]
        right_hip = landmarks[self.landmark_names.index('right_hip')]
        hip_center = np.array([(left_hip.x + right_hip.x) / 2,
                               (left_hip.y + right_hip.y) / 2])
        left_shoulder = landmarks[self.landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self.landmark_names.index('right_shoulder')]
        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                    (left_shoulder.y + right_shoulder.y) / 2])
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        max_distance = max(
            torso_size * 2.5,
            max(np.linalg.norm(
                [landmark.x - hip_center[0], landmark.y - hip_center[1]])
                for landmark in landmarks)
        )
        normalized = []
        for lm in landmarks:
            nx = (lm.x - hip_center[0]) / max_distance
            ny = (lm.y - hip_center[1]) / max_distance
            nz = lm.z / max_distance
            normalized.extend([nx, ny, nz, lm.visibility])
        return normalized

    def draw_pose(self, image, landmarks) -> np.ndarray:
        if image is None or landmarks is None:
            return None
        img = image.copy()
        h, w = img.shape[:2]
        for i, lm in enumerate(landmarks):
            # Draw with float precision for better accuracy
            cx, cy = lm.x * w, lm.y * h
            cv2.circle(img, (int(round(cx)), int(round(cy))), 4, POINT_COLOR, -1)
        for connection in self.skeleton:
            start_idx, end_idx = connection
            x1, y1 = landmarks[start_idx].x * w, landmarks[start_idx].y * h
            x2, y2 = landmarks[end_idx].x * w, landmarks[end_idx].y * h
            cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), LINE_COLOR, 2)
        return img

    def extract_from_image(self, image: np.ndarray, show: bool = False) -> Tuple[List[float], bool, np.ndarray]:
        if image is None:
            return None, False, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        if not results.pose_landmarks:
            landmarks = np.zeros(33 * 4)
            return landmarks, False, None
        landmarks = [results.pose_landmarks.landmark[i] for i in range(33)]
        normalized_data = self._normalize(landmarks)
        vis_img = self.draw_pose(image, landmarks) if show else None
        return normalized_data, True, vis_img

    def create_column_names(self) -> List[str]:
        columns = []
        for name in self.landmark_names:
            for attr in ['x', 'y', 'z', 'visibility']:
                columns.append(f'{name}_{attr}')
        return columns

class BlazePoseCOCOExtractor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=0,
            min_detection_confidence=0.3
        )
        self.landmark_names = COCO_KEYPOINT_NAMES
        self.skeleton = COCO_SKELETON  # Use COCO skeleton

    def _normalize(self, landmarks) -> List[float]:
        left_hip = landmarks[11]
        right_hip = landmarks[12]
        hip_center = np.array([(left_hip.x + right_hip.x) / 2,
                               (left_hip.y + right_hip.y) / 2])
        left_shoulder = landmarks[5]
        right_shoulder = landmarks[6]
        shoulder_center = np.array([(left_shoulder.x + right_shoulder.x) / 2,
                                    (left_shoulder.y + right_shoulder.y) / 2])
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        max_distance = max(
            torso_size * 2.5,
            max(np.linalg.norm(
                [lm.x - hip_center[0], lm.y - hip_center[1]]) for lm in landmarks)
        )
        normalized = []
        for lm in landmarks:
            nx = (lm.x - hip_center[0]) / max_distance
            ny = (lm.y - hip_center[1]) / max_distance
            nz = lm.z / max_distance
            normalized.extend([nx, ny, lm.visibility])
        return normalized

    def draw_pose(self, image, landmarks) -> np.ndarray:
        if image is None or landmarks is None:
            return None
        img = image.copy()
        h, w = img.shape[:2]
        for i, lm in enumerate(landmarks):
            if lm.visibility > 0.3:
                cx, cy = lm.x * w, lm.y * h
                cv2.circle(img, (int(round(cx)), int(round(cy))), 4, POINT_COLOR, -1)
        for start_idx, end_idx in self.skeleton:
            if (landmarks[start_idx].visibility > 0.3 and
                    landmarks[end_idx].visibility > 0.3):
                x1, y1 = landmarks[start_idx].x * w, landmarks[start_idx].y * h
                x2, y2 = landmarks[end_idx].x * w, landmarks[end_idx].y * h
                cv2.line(img, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), LINE_COLOR, 2)
        return img

    def extract_from_image(self, image: np.ndarray, show: bool = False) -> Tuple[List[float], bool, np.ndarray]:
        if image is None:
            return None, False, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(image_rgb)
        if results.pose_landmarks is None:
            landmarks = np.zeros(17 * 3)
            return landmarks, False, None
        all_landmarks = results.pose_landmarks.landmark
        landmarks = [all_landmarks[i] for i in COCO_KEYPOINTS_IDX]
        normalized_data = self._normalize(landmarks)
        vis_img = self.draw_pose(image, landmarks) if show else None
        return normalized_data, True, vis_img

    def create_column_names(self) -> List[str]:
        columns = []
        for name in self.landmark_names:
            for attr in ['x', 'y', 'visibility']:
                columns.append(f'{name}_{attr}')
        return columns

class MoveNetPoseExtractor:
    def __init__(self, model_url="https://tfhub.dev/google/movenet/singlepose/lightning/4"):
        gpus = tf.config.list_physical_devices('GPU')
        device = '/GPU:0' if gpus else '/CPU:0'
        with tf.device(device):
            self.model = hub.load(model_url)
        self.input_size = 192
        self.keypoint_names = COCO_KEYPOINT_NAMES
        self.skeleton = COCO_SKELETON  # Use COCO skeleton

    def _normalize(self, keypoints: np.ndarray) -> List[float]:
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        hip_center = (left_hip + right_hip) / 2
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        max_dist = np.max(np.linalg.norm(keypoints[:, :2] - hip_center, axis=1))
        scale = max(torso_size * 2.5, max_dist)
        norm_kp = []
        for x, y, score in keypoints:
            nx = (x - hip_center[0]) / scale
            ny = (y - hip_center[1]) / scale
            norm_kp.extend([nx, ny, score])
        return norm_kp

    def draw_pose(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        img = image.copy()
        h, w = img.shape[:2]
        keypoints_px = keypoints.copy()
        keypoints_px[:, 0] *= w
        keypoints_px[:, 1] *= h
        for i, (x, y, score) in enumerate(keypoints_px):
            if score > 0.3:
                cv2.circle(img, (int(round(x)), int(round(y))), 4, POINT_COLOR, -1)
        for i, j in self.skeleton:
            if keypoints_px[i][2] > 0.3 and keypoints_px[j][2] > 0.3:
                pt1 = (float(keypoints_px[i][0]), float(keypoints_px[i][1]))
                pt2 = (float(keypoints_px[j][0]), float(keypoints_px[j][1]))
                cv2.line(img, (int(round(pt1[0])), int(round(pt1[1])),), (int(round(pt2[0])), int(round(pt2[1]))), LINE_COLOR, 2)
        return img

    def extract_from_image(self, image: np.ndarray, show=False) -> Tuple[List[float], bool, np.ndarray]:
        if image is None:
            return None, False, None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = tf.image.resize_with_pad(tf.expand_dims(image_rgb, axis=0), self.input_size, self.input_size)
        input_img = tf.cast(input_img, dtype=tf.int32)
        outputs = self.model.signatures['serving_default'](input_img)
        keypoints = outputs['output_0'].numpy()[0, 0, :, :]
        avg_score = np.mean(keypoints[:, 2])
        if keypoints is None or keypoints.shape[0] == 0 or avg_score < 0.3:
            landmarks = np.zeros(17 * 3)
            return landmarks, False, None
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        keypoints_xyc = keypoints[:, :3]
        normalized = self._normalize(keypoints_xyc)
        vis_img = self.draw_pose(image, keypoints_xyc) if show else None
        return normalized, True, vis_img

    def create_column_names(self) -> List[str]:
        columns = []
        for name in self.keypoint_names:
            for attr in ['x', 'y', 'score']:
                columns.append(f'{name}_{attr}')
        return columns


class MoveNetPoseTFLiteExtractor:
    def __init__(self, model_path="pose_estimation\model_ske\singlepose-lightning-tflite-float16.tflite"):
        # path = kagglehub.model_download("google/movenet/tfLite/singlepose-lightning-tflite-float16")
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Input size: 192 for Lightning, 256 for Thunder
        self.input_size = self.input_details[0]['shape'][2]

        self.keypoint_names = COCO_KEYPOINT_NAMES
        self.skeleton = COCO_SKELETON

    def _normalize(self, keypoints: np.ndarray) -> List[float]:
        left_hip = keypoints[11][:2]
        right_hip = keypoints[12][:2]
        hip_center = (left_hip + right_hip) / 2
        left_shoulder = keypoints[5][:2]
        right_shoulder = keypoints[6][:2]
        shoulder_center = (left_shoulder + right_shoulder) / 2
        torso_size = np.linalg.norm(shoulder_center - hip_center)
        max_dist = np.max(np.linalg.norm(keypoints[:, :2] - hip_center, axis=1))
        scale = max(torso_size * 2.5, max_dist)
        norm_kp = []
        for x, y, score in keypoints:
            nx = (x - hip_center[0]) / scale
            ny = (y - hip_center[1]) / scale
            norm_kp.extend([nx, ny, score])
        return norm_kp

    def draw_pose(self, image: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
        if image is None:
            return None
        img = image.copy()
        h, w = img.shape[:2]
        keypoints_px = keypoints.copy()
        keypoints_px[:, 0] *= w
        keypoints_px[:, 1] *= h
        for i, (x, y, score) in enumerate(keypoints_px):
            if score > 0.3:
                cv2.circle(img, (int(round(x)), int(round(y))), 4, POINT_COLOR, -1)
        for i, j in self.skeleton:
            if keypoints_px[i][2] > 0.3 and keypoints_px[j][2] > 0.3:
                pt1 = (float(keypoints_px[i][0]), float(keypoints_px[i][1]))
                pt2 = (float(keypoints_px[j][0]), float(keypoints_px[j][1]))
                cv2.line(img,
                         (int(round(pt1[0])), int(round(pt1[1]))),
                         (int(round(pt2[0])), int(round(pt2[1]))),
                         LINE_COLOR, 2)
        return img

    def extract_from_image(self, image: np.ndarray, show=False) -> Tuple[List[float], bool, np.ndarray]:
        if image is None:
            return None, False, None

        # Preprocess: resize và convert sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = tf.image.resize_with_pad(np.expand_dims(image_rgb, axis=0),
                                       self.input_size, self.input_size)
        input_data = tf.cast(img, dtype=tf.uint8).numpy()

        # Set tensor input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get output keypoints
        keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])[0, 0, :, :]  # (17,3)

        avg_score = np.mean(keypoints[:, 2])
        if keypoints is None or keypoints.shape[0] == 0 or avg_score < 0.3:
            landmarks = np.zeros(17 * 3)
            return landmarks, False, None

        # Đảo trục x-y (MoveNet trả [y, x, score])
        keypoints[:, [0, 1]] = keypoints[:, [1, 0]]
        keypoints_xyc = keypoints[:, :3]

        normalized = self._normalize(keypoints_xyc)
        vis_img = self.draw_pose(image, keypoints_xyc) if show else None
        return normalized, True, vis_img

    def create_column_names(self) -> List[str]:
        columns = []
        for name in self.keypoint_names:
            for attr in ['x', 'y', 'score']:
                columns.append(f'{name}_{attr}')
        return columns

 
 
if __name__ == "__main__":
    IMAGE_PATH = "dataset\\tree\\tr1_0000.jpg"
    # IMAGE_PATH = "dataset\\child\\ch1_aug0_0007.jpg"
    image_np = cv2.imread(IMAGE_PATH)  # hoặc ảnh từ webcam...
    # extractor = BlazePoseExtractor()
    # extractor = BlazePoseCOCOExtractor()
    extractor = MoveNetPoseExtractor()
    keypoints_vector, success, vis_img = extractor.extract_from_image(image_np, show=True)

    print(keypoints_vector, success)

    if success and vis_img is not None:
        cv2.imshow("BlazePose Estimation", vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Không phát hiện được người trong ảnh.")
