import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm # Import thêm thư viện hiển thị tiến trình
import matplotlib.pyplot as plt # Bổ sung thư viện vẽ biểu đồ
from torch.cuda.amp import autocast, GradScaler # Kỹ thuật Mixed Precision giúp tăng tốc và giảm VRAM

from config import cfg
from models.valdnet_baseline import ValdNetBaseline
from dataset import PreprocessedVideoDataset

def get_model(model_name, cfg):
    """
    Hàm Factory giúp dễ dàng khởi tạo và thay đổi giữa nhiều model khác nhau.
    Nếu có model mới, bạn import ở trên và thêm vào đây.
    """
    if model_name == "ValdNetBaseline":
        return ValdNetBaseline(cfg)
    # elif model_name == "MyNewModel":
    #     return MyNewModel(cfg)
    else:
        raise ValueError(f"Model '{model_name}' chưa được định nghĩa trong hàm get_model!")

def train_model(model_name="ValdNetBaseline"):
    # 1. Khởi tạo Model
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = get_model(model_name, cfg).to(device)
    print(f"Bắt đầu huấn luyện mô hình [{model_name}] trên thiết bị: {device}")
    
    # 2. Hàm Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Optimizer
    # Chỉ đưa các tham số chưa bị đóng băng (requires_grad=True) vào Optimizer
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                           lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
    # Thêm Scheduler: Tự động giảm Learning Rate đi một nửa (factor=0.5) nếu Val Loss không cải thiện sau 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # 4. Load Data
    train_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    val_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("LỖI NGHIÊM TRỌNG: DataLoader trống! Vui lòng kiểm tra lại thư mục dữ liệu.")
        return

    # 5. Vòng lặp Huấn luyện
    best_val_loss = float('inf')
    epochs_no_improve = 0 # Biến đếm cho Early Stopping
    
    scaler = GradScaler() # Bộ chia tỷ lệ gradient cho FP16
    
    # 6. Data Augmentation
    # Áp dụng Augmentation cho khung hình RGB ngay trong lúc train để tăng độ đa dạng
    train_transforms = T.Compose([
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = T.Compose([
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Khởi tạo dictionary để lưu lại lịch sử độ đo
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(cfg.EPOCHS):
        # --- HUẤN LUYỆN ---
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # Thêm tqdm để vẽ thanh tiến trình cho tập Train
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}] [Train]")
        for rgb_seq, flow_seq, labels in train_bar:
            rgb_seq = rgb_seq.to(device)
            flow_seq = flow_seq.to(device)
            labels = labels.to(device)
            
            # Áp dụng Data Augmentation (chỉ trên luồng RGB)
            B, T_frames, C, H, W = rgb_seq.shape
            rgb_seq_flat = rgb_seq.reshape(B * T_frames, C, H, W)
            rgb_seq_flat = train_transforms(rgb_seq_flat)
            rgb_seq = rgb_seq_flat.reshape(B, T_frames, C, H, W)
            
            # Bắt buộc Normalize cả luồng Flow (vì cho vào mạng EfficientNet pre-trained ImageNet)
            flow_seq_flat = flow_seq.reshape(B * T_frames, C, H, W)
            flow_seq_flat = val_transforms(flow_seq_flat) # val_transforms chỉ làm Normalize nên dùng chung được
            flow_seq = flow_seq_flat.reshape(B, T_frames, C, H, W)

            optimizer.zero_grad()
            
            # Cho phép PyTorch tự động ép kiểu dữ liệu xuống FP16 để tính toán nhanh hơn
            with autocast():
                predictions = model(rgb_seq, flow_seq)
                
                predictions = predictions.reshape(-1)
                labels = labels.reshape(-1)

                # Áp dụng Label Smoothing (0.1) để giảm sự tự tin thái quá của mô hình
                # Biến nhãn 1.0 -> 0.95 và nhãn 0.0 -> 0.05
                smoothed_labels = labels * 0.9 + 0.05
                loss = criterion(predictions, smoothed_labels)
            
            # Lan truyền ngược sử dụng GradScalerc
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            predicted = (predictions > 0.0).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            # Cập nhật loss thời gian thực lên thanh tiến trình
            train_bar.set_postfix(loss=loss.item())
            
        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # --- ĐÁNH GIÁ ---
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{cfg.EPOCHS}] [Val]  ")
            for rgb_seq, flow_seq, labels in val_bar:
                rgb_seq = rgb_seq.to(device)
                flow_seq = flow_seq.to(device)
                labels = labels.to(device)
                
                # Áp dụng Normalize cho tập Val
                B, T_frames, C, H, W = rgb_seq.shape
                rgb_seq_flat = rgb_seq.reshape(B * T_frames, C, H, W)
                rgb_seq_flat = val_transforms(rgb_seq_flat)
                rgb_seq = rgb_seq_flat.reshape(B, T_frames, C, H, W)
                
                # Áp dụng Normalize cho tập Val của luồng Flow
                flow_seq_flat = flow_seq.reshape(B * T_frames, C, H, W)
                flow_seq_flat = val_transforms(flow_seq_flat)
                flow_seq = flow_seq_flat.reshape(B, T_frames, C, H, W)

                # Cũng dùng Mixed Precision khi đánh giá Validation
                with autocast():
                    predictions = model(rgb_seq, flow_seq)
                    predictions = predictions.reshape(-1)
                    labels = labels.reshape(-1)

                    # Áp dụng Label Smoothing tương tự cho quá trình tính Val Loss
                    smoothed_labels = labels * 0.9 + 0.05
                    loss = criterion(predictions, smoothed_labels)
                
                val_loss += loss.item()
                predicted = (predictions > 0.0).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
            
        # Lưu thông số vào history để vẽ biểu đồ sau khi train xong
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n=> Kết quả Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
        # Cập nhật Learning Rate Scheduler dựa trên Val Loss
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("   [+] Đã lưu model tốt nhất (best_model.pth)!\n")
        else:
            epochs_no_improve += 1
            print(f"   [-] Val Loss không giảm ({epochs_no_improve}/{cfg.PATIENCE})\n")
            
            # Kiểm tra Early Stopping
            if epochs_no_improve >= cfg.PATIENCE:
                print(f"Báo động: Đã dừng sớm (Early Stopping) tại Epoch {epoch+1} để tránh Overfitting!")
                break

    # --- KẾT THÚC HUẤN LUYỆN ---
    # Tự động load lại model tốt nhất vào bộ nhớ thay vì giữ model của epoch cuối cùng
    model.load_state_dict(torch.load("best_model.pth"))
    print("\n[+] Đã nạp lại 'best_model.pth' vào RAM. Sẵn sàng cho việc Inference/Testing!")

    # --- VẼ BIỂU ĐỒ KẾT QUẢ ---
    print("\nĐang vẽ biểu đồ quá trình huấn luyện...")
    plt.figure(figsize=(12, 5))
    
    # 1. Biểu đồ Loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='o')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 2. Biểu đồ Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='o')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    print("Đã lưu biểu đồ thành công tại file: 'training_history.png'")
    plt.close()

if __name__ == "__main__":
    # Đổi tên model ở đây khi bạn muốn train model khác
    train_model(model_name="ValdNetBaseline")