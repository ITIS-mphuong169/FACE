import cv2
import numpy as np
import joblib
import time

from pose_estimation.blazepose_movenet_extractors import MoveNetPoseTFLiteExtractor

# Load SVM model đã train
svm_model = joblib.load("saved_models/SVM.joblib")

# Lấy danh sách nhãn (nếu cần, có thể load từ file hoặc hardcode)
# Ví dụ: class_list = ['tree', 'child', ...]
# Nếu có file class_list.json thì load, nếu không thì hardcode
class_list = ["Cay cau","Em be","Cho up mat","Tam van thap","Ngon nui 1","Ngon nui 2","Tam van","Cui nguoi ve phia truoc","Cai cay","Tam giac","Chien binh 1","Chien binh 2"]

extractor = MoveNetPoseTFLiteExtractor()

cap = cv2.VideoCapture(0)

# Lấy thông số video để lưu (sửa lại thành vuông)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
min_dim = min(frame_width, frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 25  # fallback nếu không lấy được

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_pred_video_phat.mp4', fourcc, fps, (min_dim, min_dim))

frame_count = 0
fps_list = []
prev_time = time.time()

def crop_center_square(frame):
    h, w = frame.shape[:2]
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    return frame[start_y:start_y+min_dim, start_x:start_x+min_dim]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Crop frame thành hình vuông
    frame_square = crop_center_square(frame)

    # Trích xuất keypoints từ frame vuông
    keypoints_vector, success, vis_img = extractor.extract_from_image(frame_square, show=True)
    print(keypoints_vector)
    if success:
        # Dự đoán nhãn
        X = np.array(keypoints_vector).reshape(1, -1)
        pred = int(svm_model.predict(X)[0])

        # Nếu có class_list thì chuyển số sang tên
        if class_list and isinstance(pred, int) and pred < len(class_list):
            label = class_list[pred]

        # Hiển thị nhãn lên hình
        cv2.putText(vis_img, f'{label}', (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        display_img = vis_img
    else:
        display_img = frame_square

    # Tính FPS
    curr_time = time.time()
    fps_now = 1.0 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps_now)
    frame_count += 1

    cv2.putText(display_img, f'FPS: {fps_now:.2f}', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    # Hiển thị và ghi video
    cv2.imshow("MoveNet Skeleton + Prediction", display_img)
    out.write(display_img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC để thoát
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Tính FPS trung bình
if fps_list:
    avg_fps = sum(fps_list) / len(fps_list)
    print(f"Trung bình FPS: {avg_fps:.2f}")
else:
    print("Không có frame nào được xử lý.")
