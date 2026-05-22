import os
import shutil
import random

# ==========================================
# 1. CẤU HÌNH ĐƯỜNG DẪN VÀ TỶ LỆ
# ==========================================
RAW_DATA_DIR = "D:\\Violent_Detection\\dataset\\data"          # Thư mục đang chứa hỗn lốn video của bạn
OUTPUT_DIR = "D:\\Violent_Detection\\my_dataset"      # Thư mục đích sẽ chứa data đã chia chuẩn
TRAIN_RATIO = 0.8                # 80% cho Train, 20% cho Validation

# ==========================================
# 2. HÀM KIỂM TRA NHÃN DỰA VÀO TÊN FILE
# ==========================================
# BẠN CẦN SỬA HÀM NÀY TÙY THEO CÁCH ĐẶT TÊN CỦA DATASET BẠN TẢI VỀ
def is_violence(filename):
    name = filename.lower()
    # Chỉ giữ lại các từ khóa đại diện cho video bạo lực
    if "fi" in name or "fight" in name or "v_" in name or "violence" in name:
        return True
    return False

# ==========================================
# 3. KỊCH BẢN CHÍNH
# ==========================================
def main():
    # Lấy danh sách tất cả các video trong thư mục gốc
    all_videos = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith(('.mp4', '.avi', '.mkv'))]
    
    # Phân loại list video thành 2 mảng riêng biệt
    violence_vids = [f for f in all_videos if is_violence(f)]
    normal_vids = [f for f in all_videos if not is_violence(f)]
    
    print(f"Tổng số video: {len(all_videos)}")
    print(f" - Bạo lực: {len(violence_vids)} video")
    print(f" - Bình thường: {len(normal_vids)} video")
    
    # Trộn ngẫu nhiên để đảm bảo tính khách quan khi train
    random.seed(42) # Cố định seed để nếu chạy lại vẫn ra kết quả giống nhau
    random.shuffle(violence_vids)
    random.shuffle(normal_vids)
    
    # Tính toán điểm cắt (Split index)
    v_split_idx = int(len(violence_vids) * TRAIN_RATIO)
    n_split_idx = int(len(normal_vids) * TRAIN_RATIO)
    
    # Đóng gói thành Dictionary để dễ xử lý bằng vòng lặp
    splits = {
        "train": {
            "violence": violence_vids[:v_split_idx],
            "non_violence": normal_vids[:n_split_idx]
        },
        "val": {
            "violence": violence_vids[v_split_idx:],
            "non_violence": normal_vids[n_split_idx:]
        }
    }
    
    # Bắt đầu quá trình tạo thư mục và COPY file
    for phase in ["train", "val"]:
        for category in ["violence", "non_violence"]:
            # Tạo đường dẫn thư mục đích (Ví dụ: my_dataset/train/violence)
            target_folder = os.path.join(OUTPUT_DIR, phase, category)
            os.makedirs(target_folder, exist_ok=True)
            
            # Lấy danh sách video tương ứng
            videos_to_copy = splits[phase][category]
            
            print(f"Đang copy {len(videos_to_copy)} video vào {target_folder}...")
            
            for vid in videos_to_copy:
                src_path = os.path.join(RAW_DATA_DIR, vid)
                dst_path = os.path.join(target_folder, vid)
                
                # Khuyên dùng copy (thay vì move) để giữ lại bản gốc dự phòng
                shutil.copy(src_path, dst_path)

    print("Hoàn tất! Dữ liệu đã sẵn sàng để đưa vào Dataset của PyTorch.")

if __name__ == "__main__":
    main()