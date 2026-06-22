import os
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.amp import autocast

from config import Config as cfg
from models.valdnet_baseline import TwoStreamBiLSTMModel
from inference import preprocess_video

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang sử dụng thiết bị: {device}")

    # 1. Khởi tạo model và load trọng số tốt nhất
    model = TwoStreamBiLSTMModel(
        num_classes=1, 
        rnn_hidden_size=cfg.LSTM_HIDDEN_SIZE,
        num_rnn_layers=cfg.LSTM_LAYERS,
        freeze_backbone=False
    ).to(device)
    
    model_path = "best_model.pth"
    if not os.path.exists(model_path):
        print(f"Không tìm thấy file trọng số {model_path}. Vui lòng huấn luyện mô hình trước!")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # 2. Các hàm chuẩn hóa (Giống quá trình train)
    normalize_rgb = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    normalize_flow = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    test_dir = r"D:\Violent_Detection\my_dataset\test"
    categories = {
        "non_violence": 0,
        "violence": 1
    }
    
    all_labels = []
    all_preds = []

    print("\nBắt đầu đánh giá mô hình trên tập Test...")
    
    # 3. Duyệt qua từng video trong tập test
    for category, label in categories.items():
        folder_path = os.path.join(test_dir, category)
        if not os.path.exists(folder_path):
            print(f"Bỏ qua: Không tìm thấy thư mục {folder_path}")
            continue
            
        videos = [f for f in os.listdir(folder_path) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        
        for vid_name in tqdm(videos, desc=f"Testing {category}"):
            vid_path = os.path.join(folder_path, vid_name)
            
            # Tiền xử lý RGB & Optical Flow
            rgb_seq, flow_seq = preprocess_video(vid_path)
            
            if rgb_seq is None or flow_seq is None:
                continue # Bỏ qua video bị lỗi hoặc không đủ số frame
                
            rgb_seq = rgb_seq.to(device)
            flow_seq = flow_seq.to(device)
            
            B, T, C, H, W = rgb_seq.shape
            rgb_seq = normalize_rgb(rgb_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
            flow_seq = normalize_flow(flow_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
            
            # Dự đoán
            with torch.no_grad():
                with autocast(device.type, enabled=(device.type == "cuda")):
                    outputs = model(rgb_seq, flow_seq)
                    prob = torch.sigmoid(outputs).item()
                    pred = 1 if prob > 0.5 else 0
                    
            all_labels.append(label)
            all_preds.append(pred)

    if len(all_labels) == 0:
        print("Không có video nào được đánh giá. Hãy kiểm tra lại dữ liệu!")
        return

    # 4. Tính toán và in báo cáo
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Non-Violence", "Violence"])

    print("\n" + "="*55)
    print("      KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST (300 VIDEOS)      ")
    print("="*55)
    print(f"Độ chính xác (Accuracy) : {acc * 100:.2f}%")
    print(f"F1-Score                : {f1:.4f}")
    print("\n[Báo cáo chi tiết]")
    print(report)
    print("Confusion Matrix")
    print(cm)
    print("="*55)

if __name__ == "__main__":
    main()