import os
import cv2
import numpy as np
import torch
from config import cfg
from tqdm import tqdm
from utils import compute_farneback_flow

def process_single_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < cfg.T_FRAMES:
        cap.release() # Vá lỗi rò rỉ RAM nếu video lỗi bị return sớm
        return False 
        
    # Tạo set() các index cần lấy để tra cứu O(1)
    target_indices = set(np.linspace(0, total_frames - 2, cfg.T_FRAMES).astype(int))
    
    # 1. TIỀN CẤP PHÁT BỘ NHỚ (Pre-allocation) - Siêu nhanh và tiết kiệm RAM
    H, W = cfg.IMAGE_SIZE, cfg.IMAGE_SIZE
    np_rgb = np.empty((cfg.T_FRAMES, H, W, 3), dtype=np.uint8)
    np_flow = np.empty((cfg.T_FRAMES, H, W, 3), dtype=np.uint8)
    
    frame_idx = 0
    saved_idx = 0
    
    # Đọc frame đầu tiên làm mốc
    ret, prev_frame = cap.read()
    if not ret:
        cap.release()
        return False
        
    # 2. ĐỌC TUẦN TỰ (Sequential Read) - Loại bỏ hoàn toàn cap.set()
    while True:
        ret, curr_frame = cap.read()
        if not ret:
            break
            
        if frame_idx in target_indices:
            # Resize
            prev_resized = cv2.resize(prev_frame, (W, H))
            curr_resized = cv2.resize(curr_frame, (W, H))
            
            # Tính Flow
            flow = compute_farneback_flow(prev_resized, curr_resized)
            
            # Đổ trực tiếp vào vùng nhớ đã cấp phát
            np_rgb[saved_idx] = cv2.cvtColor(prev_resized, cv2.COLOR_BGR2RGB)
            np_flow[saved_idx] = flow
            
            saved_idx += 1
            if saved_idx == cfg.T_FRAMES:
                break # Đủ frame thì thoát sớm, không cần đọc hết phần video dư thừa
                
        prev_frame = curr_frame
        frame_idx += 1
        
    cap.release()
    
    if saved_idx < cfg.T_FRAMES:
        return False
        
    # Chuyển đổi tensor chuẩn xác hơn, tránh duplicate data từ numpy sang pytorch
    tensor_rgb = torch.from_numpy(np_rgb).permute(0, 3, 1, 2).float() / 255.0
    tensor_flow = torch.from_numpy(np_flow).permute(0, 3, 1, 2).float() / 255.0
    
    torch.save({'rgb': tensor_rgb, 'flow': tensor_flow}, output_path)
    return True

def main():
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for phase in ['train', 'val']:
        for category in ['violence', 'non_violence']:
            in_folder = os.path.join(cfg.DATA_DIR, phase, category)
            out_folder = os.path.join(cfg.PROCESSED_DATA_DIR, phase, category)
            
            if not os.path.exists(in_folder):
                continue
                
            os.makedirs(out_folder, exist_ok=True)
            videos = [f for f in os.listdir(in_folder) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            
            for vid_name in tqdm(videos, desc=f"{phase}/{category}"):
                vid_path = os.path.join(in_folder, vid_name)
                out_name = os.path.splitext(vid_name)[0] + '.pt'
                out_path = os.path.join(out_folder, out_name)
                
                if os.path.exists(out_path):
                    total_skipped += 1
                    continue

                if process_single_video(vid_path, out_path):
                    total_processed += 1
                else:
                    total_errors += 1

    print("\n=== HOÀN TẤT TIỀN XỬ LÝ ===")
    print(f"Thành công: {total_processed} | Bỏ qua: {total_skipped} | Lỗi: {total_errors}")

if __name__ == "__main__":
    main()  