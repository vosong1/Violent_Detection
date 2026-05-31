import cv2
import numpy as np

def compute_farneback_flow(prev_frame, next_frame):
    prvs = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # Tính Optical Flow với tham số chuẩn
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    
    # Khởi tạo ma trận HSV
    hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
    
    # Kênh Hue: Định vị HƯỚNG di chuyển (Góc)
    hsv[..., 0] = np.clip(ang / 2, 0, 179).astype(np.uint8)
    
    # Kênh Saturation: Đặt tối đa (255) để màu sắc rõ ràng nhất có thể
    hsv[..., 1] = 255
    
    # Kênh Value: CƯỜNG ĐỘ di chuyển
    # Dùng cv2.normalize để tự động co giãn giá trị mag vào khoảng [0, 255]
    # Cách này dẹp bỏ hoàn toàn hiện tượng "cháy sáng" (blowout) thành mây xanh
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    # Trả về RGB (Phông nền tĩnh sẽ có màu Đen, vùng chuyển động sẽ có màu rực rỡ)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)