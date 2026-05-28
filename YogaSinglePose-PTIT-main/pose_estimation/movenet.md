note : python >=3.12
pip install movenet-pre-processor
pip install tensorflow tensorflow-hub opencv-python numpy
pip install  tensorflow-hub 

python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
pip uninstall tensorflow
pip install tensorflow[and-cuda]
pip install "tensorflow[and-cuda]==2.19.0"
