import cv2
import numpy as np
import torch

def compute_farneback_flow(prev_frame, next_frame):
    """
    Tính Dense Optical Flow bằng Farneback.
    Input: prev_frame, next_frame (ảnh BGR của OpenCV)
    Output: Ảnh Optical Flow 3 kênh (RGB numpy array)
    """
    # Chuyển sang ảnh xám
    prvs = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Tính Optical Flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    # Chuyển đổi vector (dx, dy) sang ảnh màu RGB (HSV -> RGB)
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return flow_rgb