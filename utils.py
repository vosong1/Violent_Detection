import cv2
import numpy as np

def compute_farneback_flow(prev_frame, next_frame):
    prvs = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    nxt = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Tính Dense Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prvs, nxt, None, 
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=True)
    
    # 2. KHÔI PHỤC HÌNH DÁNG NGƯỜI ĐẶC (Silhouette)
    # Dùng Gaussian Blur để làm mượt các vector di chuyển, giúp các vùng chuyển động 
    # lan tỏa và dính liền vào nhau thành một khối người hoàn chỉnh.
    mag = cv2.GaussianBlur(mag, (5, 5), 0)
    
    # Lọc bỏ các nhiễu nền hoặc rung lắc nhẹ của camera (Pixel di chuyển < 1.5)
    mag[mag < 1.5] = 0 
    
    # 3. TẠO MÀU SẮC TỐI ƯU CHO CNN
    hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
    
    # Kênh Hue: Màu sắc biểu diễn hướng di chuyển
    hsv[..., 0] = np.clip(ang / 2, 0, 179).astype(np.uint8)
    
    # Kênh Saturation: Biểu diễn độ mạnh của chuyển động
    # Nhân mag với 25 để chuyển động nhỏ nhất cũng lên màu rực rỡ, lấp đầy hình dáng người
    hsv[..., 1] = np.clip(mag * 25, 0, 255).astype(np.uint8)
    
    # Kênh Value: Độ sáng
    # Đặt TOÀN BỘ bằng 255. 
    # Mẹo ở đây là: Vùng có mag=0 (nền) sẽ có Saturation=0 -> Ép thành màu TRẮNG TINH.
    # Vùng có mag>0 (người) sẽ có Saturation cao -> Hiện màu sắc rõ nét trên nền trắng.
    hsv[..., 2] = 255
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)