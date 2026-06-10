import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from config import Config as cfg
from models.valdnet_baseline import TwoStreamBiLSTMModel
from utils import compute_farneback_flow

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < cfg.T_FRAMES:
        print(f"Video quá ngắn ({total_frames} frames). Cần ít nhất {cfg.T_FRAMES} frames.")
        cap.release()
        return None, None
        
    # Tính toán các frame index sẽ được trích xuất
    target_indices = set(np.linspace(0, total_frames - 2, cfg.T_FRAMES).astype(int))
    
    H, W = cfg.IMAGE_SIZE, cfg.IMAGE_SIZE
    np_rgb = np.empty((cfg.T_FRAMES, H, W, 3), dtype=np.uint8)
    np_flow = np.empty((cfg.T_FRAMES, H, W, 3), dtype=np.uint8)
    
    frame_idx = 0
    saved_idx = 0
    
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return None, None
        
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        if frame_idx in target_indices:
            prev_resized = cv2.resize(prev_frame, (W, H))
            curr_resized = cv2.resize(curr_frame, (W, H))
            
            # Tính Optical Flow
            flow = compute_farneback_flow(prev_resized, curr_resized)
            
            np_rgb[saved_idx] = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2RGB)
            np_flow[saved_idx] = flow
            
            saved_idx += 1
            if saved_idx == cfg.T_FRAMES:
                break
                
        prev_frame = curr_frame
        frame_idx += 1
        
    cap.release()
    
    if saved_idx < cfg.T_FRAMES:
        return None, None
        
    tensor_rgb = torch.from_numpy(np_rgb).permute(0, 3, 1, 2).float() / 255.0
    tensor_flow = torch.from_numpy(np_flow).permute(0, 3, 1, 2).float() / 255.0
    
    # Thêm batch dimension -> shape: (1, T, C, H, W)
    return tensor_rgb.unsqueeze(0), tensor_flow.unsqueeze(0)

def main(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Khởi tạo model và load trọng số
    model = TwoStreamBiLSTMModel(
        num_classes=1, 
        rnn_hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_rnn_layers=cfg.LSTM_LAYERS,
        freeze_backbone=True
    ).to(device)
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file trọng số {model_path}. Vui lòng train model trước!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Quan trọng: Chuyển về chế độ đánh giá để tắt Dropout

    # 2. Tiền xử lý dữ liệu
    print(f"Đang trích xuất đặc trưng từ video: {video_path}...")
    rgb_seq, flow_seq = preprocess_video(video_path)
    
    if rgb_seq is None or flow_seq is None:
        print("Lỗi: Không thể tiền xử lý video.")
        return

    rgb_seq = rgb_seq.to(device)
    flow_seq = flow_seq.to(device)

    # 3. Chuẩn hóa phân phối giống lúc Train (Rất quan trọng)
    normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize_flow = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    B, T, C, H, W = rgb_seq.shape
    rgb_seq = normalize_rgb(rgb_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
    flow_seq = normalize_flow(flow_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)

    # 4. Dự đoán
    print("Đang dự đoán...")
    with torch.no_grad():
        outputs = model(rgb_seq, flow_seq)
        probability = torch.sigmoid(outputs).item()
        
    prediction = "Violence (Bạo lực)" if probability > 0.5 else "Non-Violence (Bình thường)"
    confidence = probability if probability > 0.5 else 1 - probability
    
    print("\n" + "="*50)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Kết quả dự đoán : {prediction}")
    print(f"Độ tự tin       : {confidence * 100:.2f}%")
    print("="*50 + "\n")

if __name__ == "__main__":
    # Thay đổi đường dẫn này trỏ tới video bạn muốn test
    TEST_VIDEO_PATH = r"test_videos\sample_fight.mp4" 
    
    if os.path.exists(TEST_VIDEO_PATH):
        main(TEST_VIDEO_PATH)
    else:
        print(f"Lỗi: Không tìm thấy video tại đường dẫn {TEST_VIDEO_PATH}")