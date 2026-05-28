https://mmpose.readthedocs.io/en/latest/installation.html
✅ Cài đặt MMPose chạy bằng CPU từ đầu (chuẩn hóa môi trường)
🔹 Bước 1: (Tuỳ chọn nhưng nên làm) – Tạo môi trường mới sạch:
pip cache purge

conda create -n mmposepython=3.9.21 -y
conda activate mmpose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116

🔹 Bước 2: Cài OpenMIM (quản lý gói của OpenMMLab)
pip install openmim

🔹 Bước 3: Cài đặt MMEngine và MMCV (bằng CPU)
mim install mmengine
mim install "mmcv>=2.0.1,<2.2.0"
<<<<<<< HEAD
cái này dùng cho vitpose :
mim install "mmpretrain>=1.0.0"

=======
>>>>>>> 9a0eedad13ea6dd0c4ed2d2b2285dcc188215644
📌 Lưu ý: Nếu bạn đang cài CPU, mim sẽ tự động chọn đúng bản mmcv không có CUDA.

🔹 Bước 4: Cài đặt MMPose
mim install "mmpose>=1.0.0"

🔹 Bước 5 (nếu cần): Cài thêm MMDetection (nếu dùng detect người trước khi pose)
mim install "mmdet>=3.1.0,<3.3.0"

✅ Kiểm tra nhanh sau cài đặt
pip list | findstr "mmcv mmengine mmpose mmdet"
(mmpose) E:\mmpose>pip list | findstr "mmcv mmengine mmpose mmdet"
mmcv                          2.1.0
mmdet                         3.2.0
mmengine                      0.10.7
mmpose                        1.3.2                e:\mmpose

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu116
pip install openmim
mim install mmengine==0.10.7
mim install mmcv==2.1.0
mim install mmpose==1.3.2
mim install mmdet==3.2.0


train machine learning 
pip install seaborn
pip install xgboost
pip install lightgbm
pip install scikit-learn
pip install matplotlib

pip show seaborn 0.13.2
pip show xgboost 2.1.4
pip show lightgbm 4.6.0
pip show scikit-learn 1.6.1
pip show matplotlib 3.9.4