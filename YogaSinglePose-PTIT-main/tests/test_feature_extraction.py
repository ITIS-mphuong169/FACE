import unittest
import numpy as np
from pose_estimation.blazepose_test_image import BlazePoseExtractor, BlazePoseCOCOExtractor

class TestFeatureExtraction(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((256, 256, 3), dtype=np.uint8)
        self.extractor = BlazePoseExtractor()
        self.coco_extractor = BlazePoseCOCOExtractor()

    def test_blazepose_column_names(self):
        columns = self.extractor.create_column_names()
        self.assertEqual(len(columns), 33*4)
        self.assertTrue(all(isinstance(col, str) for col in columns))

    def test_blazepose_coco_column_names(self):
        columns = self.coco_extractor.create_column_names()
        self.assertEqual(len(columns), 17*3)
        self.assertTrue(all(isinstance(col, str) for col in columns))

if __name__ == "__main__":
    unittest.main()
