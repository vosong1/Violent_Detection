import os
import cv2
import numpy as np
import torch
from config import cfg
from utils import compute_farneback_flow # Hàm tính Flow ở phần trước

def process_single_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < 2:
        return False # Bỏ qua video quá ngắn
        
    # --- CHIẾN LƯỢC: LẤY MẪU ĐỀU (UNIFORM SAMPLING) ---
    # Chọn ra T_FRAMES chỉ số khung hình cách đều nhau
    indices = np.linspace(0, total_frames - 2, cfg.T_FRAMES).astype(int)
    
    rgb_frames = []
    flow_frames = []
    
    for idx in indices:
        # Lấy khung hình hiện tại
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, frame1 = cap.read()
        
        # Lấy khung hình tiếp theo ngay sau đó để tính chuyển động
        # (Bỏ cap.set() vì sau lệnh read() ở trên, con trỏ đã tự tăng lên idx + 1)
        ret2, frame2 = cap.read()
        
        if not ret1 or not ret2:
            continue
            
        # Resize ảnh về kích chuẩn của EfficientNet (224x224)
        frame1 = cv2.resize(frame1, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        frame2 = cv2.resize(frame2, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
        
        # Chuyển BGR sang RGB cho frame1 (Các mô hình CNN thường yêu cầu RGB)
        frame1_rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
        
        # Tính Optical Flow
        flow = compute_farneback_flow(frame1, frame2)
        
        rgb_frames.append(frame1_rgb)
        flow_frames.append(flow)
        
    cap.release()
    
    # Kiểm tra xem có lấy đủ T khung hình không
    if len(rgb_frames) < cfg.T_FRAMES:
        return False
        
    # Chuyển List thành Numpy (Shape: T, H, W, C)
    np_rgb = np.array(rgb_frames)
    np_flow = np.array(flow_frames)
    
    # Đổi chuẩn Numpy sang PyTorch Tensor (Shape: T, C, H, W)
    # Đồng thời chuẩn hóa giá trị pixel từ [0-255] về [0-1]
    tensor_rgb = torch.tensor(np_rgb).permute(0, 3, 1, 2).float() / 255.0
    tensor_flow = torch.tensor(np_flow).permute(0, 3, 1, 2).float() / 255.0
    
    # Lưu xuống ổ cứng
    torch.save({'rgb': tensor_rgb, 'flow': tensor_flow}, output_path)
    return True

def main():
    for phase in ['train', 'val']:
        for category in ['violence', 'non_violence']:
            in_folder = os.path.join(cfg.DATA_DIR, phase, category)
            out_folder = os.path.join(cfg.PROCESSED_DATA_DIR, phase, category)
            
            # Kiểm tra xem thư mục đầu vào có tồn tại không để tránh lỗi FileNotFoundError
            if not os.path.exists(in_folder):
                print(f"Cảnh báo: Thư mục {in_folder} không tồn tại, bỏ qua.")
                continue
                
            os.makedirs(out_folder, exist_ok=True)
            
            # Lọc để chỉ lấy các file video
            videos = [f for f in os.listdir(in_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            print(f"\nĐang xử lý {len(videos)} video tại {in_folder}...")
            
            for vid_name in videos:
                vid_path = os.path.join(in_folder, vid_name)
                # Đổi đuôi .mp4 thành .pt
                out_name = vid_name.rsplit('.', 1)[0] + '.pt'
                out_path = os.path.join(out_folder, out_name)
                
                # Kiểm tra xem file đã tồn tại chưa để hỗ trợ chạy tiếp (resume) khi gián đoạn
                if os.path.exists(out_path):
                    print(f"  -> Bỏ qua (đã tồn tại): {out_name}")
                    continue

                if process_single_video(vid_path, out_path):
                    print(f"  -> Đã lưu: {out_name}")
                else:
                    print(f"  -> Lỗi/Bỏ qua: {vid_name}")

if __name__ == "__main__":
    main()