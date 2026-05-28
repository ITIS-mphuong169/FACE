import unittest

class TestFeedbackDisplay(unittest.TestCase):
    def test_feedback_message(self):
        # Dummy test for feedback display
        feedback = "Training completed successfully!"
        self.assertIn("successfully", feedback)

if __name__ == "__main__":
    unittest.main()
