import unittest
import numpy as np
from pose_estimation.ml_classifier import (
    train_with_bayes_opt,
    knn_params,
    load_data_ml
)
from sklearn.neighbors import KNeighborsClassifier

class TestMLClassifier(unittest.TestCase):
    def setUp(self):
        # Tạo dữ liệu giả
        self.x = np.random.rand(100, 10)
        self.y = np.random.randint(0, 3, 100)

    def test_train_with_bayes_opt_knn(self):
        model = KNeighborsClassifier()
        # Chạy thử với dữ liệu nhỏ
        try:
            train_with_bayes_opt(model, knn_params, self.x, self.y, model_name="KNN")
        except Exception as e:
            self.fail(f"train_with_bayes_opt raised Exception: {e}")

if __name__ == "__main__":
    unittest.main()
