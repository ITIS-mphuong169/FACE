import pandas as pd
import numpy as np

def load_data(path_csv):
    """
    Load dữ liệu từ CSV và trả về:
    x: numpy array 2D (features)
    y: numpy array 1D (labels đã factorize)
    hpe_status: numpy array 1D (nếu có), None nếu không có
    """
    df = pd.read_csv(path_csv)

    # Factorize nhãn
    y, _ = df['pose_class'].factorize()
    y = y.astype(int)

    # Tách hpe_status nếu có
    if 'hpe_status' in df.columns:
        hpe_status = df['hpe_status'].to_numpy().reshape(-1)
        x = df.drop(columns=['pose_class', 'hpe_status']).to_numpy()
    else:
        hpe_status = None
        x = df.drop(columns=['pose_class']).to_numpy()

    return x, y, hpe_status


def load_data_hpe_true(path_csv):
    """
    Load dữ liệu từ CSV, chỉ giữ các mẫu có hpe_status == 1.
    Trả về:
    x: numpy array 2D (features)
    y: numpy array 1D (labels đã factorize)
    hpe_status: numpy array 1D (chỉ có giá trị 1)
    """
    df = pd.read_csv(path_csv)

    if 'hpe_status' not in df.columns:
        raise ValueError("File CSV không có cột 'hpe_status'")

    # Lọc các mẫu hpe_status == 1
    df = df[df['hpe_status'] == 1]

    # Factorize nhãn
    y, _ = df['pose_class'].factorize()
    y = y.astype(int)

    # Lấy x và hpe_status
    hpe_status = df['hpe_status'].to_numpy().reshape(-1)
    x = df.drop(columns=['pose_class', 'hpe_status']).to_numpy()

    return x, y, hpe_status