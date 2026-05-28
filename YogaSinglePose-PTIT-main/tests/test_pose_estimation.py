import unittest
import cv2
import numpy as np
from pose_estimation.blazepose_test_image import BlazePoseExtractor, BlazePoseCOCOExtractor
from pose_estimation.mmpose_test_image import COCOPoseExtractor
from pose_estimation.movenet_test_image import MoveNetPoseExtractor

class TestPoseEstimation(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)

    def test_blazepose_extractor(self):
        extractor = BlazePoseExtractor()
        keypoints, success, _ = extractor.extract_from_image(self.image)
        self.assertEqual(len(keypoints), 33*4)
        self.assertIsInstance(success, bool)

    def test_blazepose_coco_extractor(self):
        extractor = BlazePoseCOCOExtractor()
        keypoints, success, _ = extractor.extract_from_image(self.image)
        self.assertEqual(len(keypoints), 17*3)
        self.assertIsInstance(success, bool)

    def test_mmpose_extractor(self):
        # Dummy config and checkpoint paths
        config = 'model_ske/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
        checkpoint = 'model_ske/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
        extractor = COCOPoseExtractor(config, checkpoint, device='cpu')
        keypoints, success, _ = extractor.extract_from_image(self.image)
        self.assertEqual(len(keypoints), 17*3)
        self.assertIsInstance(success, bool)

    def test_movenet_extractor(self):
        extractor = MoveNetPoseExtractor()
        keypoints, success, _ = extractor.extract_from_image(self.image)
        self.assertEqual(len(keypoints), 17*3)
        self.assertIsInstance(success, bool)

if __name__ == "__main__":
    unittest.main()
