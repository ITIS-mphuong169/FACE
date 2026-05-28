# pose_estimation

This folder contains pose estimation models and scripts:

- `blazepose_test_image.py`, `blazepose_run_output_csv.py`: BlazePose keypoint extraction and CSV output
- `mmpose_test_image.py`, `mmpose_run_output_csv.py`: MMPose keypoint extraction and CSV output
- `movenet_test_image.py`, `movenet_run_output_csv.py`: MoveNet keypoint extraction and CSV output

## Usage

Import the relevant extractor class from the package and use the run scripts to process images and export keypoints to CSV.

## Requirements
- mediapipe
- mmpose
- tensorflow
- tensorflow_hub
- opencv-python
- numpy
- pandas
