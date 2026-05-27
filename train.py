import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm # Import thêm thư viện hiển thị tiến trình
from torch.cuda.amp import autocast, GradScaler # Kỹ thuật Mixed Precision giúp tăng tốc và giảm VRAM

from config import cfg
from models.valdnet_baseline import ValdNetBaseline
from dataset import PreprocessedVideoDataset

def train_model():
    # 1. Khởi tạo Model
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = ValdNetBaseline(cfg).to(device)
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    # 2. Hàm Loss
    criterion = nn.BCEWithLogitsLoss()
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    
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
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
    ])
    
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
            rgb_seq_flat = rgb_seq.view(B * T_frames, C, H, W)
            rgb_seq_flat = train_transforms(rgb_seq_flat)
            rgb_seq = rgb_seq_flat.view(B, T_frames, C, H, W)
            
            optimizer.zero_grad()
            
            # Cho phép PyTorch tự động ép kiểu dữ liệu xuống FP16 để tính toán nhanh hơn
            with autocast():
                predictions = model(rgb_seq, flow_seq)
                
                predictions = predictions.view(-1)
                labels = labels.view(-1)

                loss = criterion(predictions, labels)
            
            # Lan truyền ngược sử dụng GradScaler
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
                
                # Cũng dùng Mixed Precision khi đánh giá Validation
                with autocast():
                    predictions = model(rgb_seq, flow_seq)
                    predictions = predictions.view(-1)
                    labels = labels.view(-1)

                    loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted = (predictions > 0.0).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
            
        print(f"\n=> Kết quả Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
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

if __name__ == "__main__":
    train_model()