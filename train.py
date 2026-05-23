import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # Import thêm thư viện hiển thị tiến trình

from config import cfg
from models.valdnet_baseline import ValdNetBaseline
from dataset import PreprocessedVideoDataset

def train_model():
    # 1. Khởi tạo Model
    device = torch.device(cfg.DEVICE if torch.cuda.is_available() else "cpu")
    model = ValdNetBaseline(cfg).to(device)
    print(f"Bắt đầu huấn luyện trên thiết bị: {device}")
    
    # 2. Hàm Loss
    criterion = nn.BCELoss()
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # 4. Load Data
    train_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    val_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    if len(train_loader) == 0 or len(val_loader) == 0:
        print("LỖI NGHIÊM TRỌNG: DataLoader trống! Vui lòng kiểm tra lại thư mục dữ liệu.")
        return

    # 5. Vòng lặp Huấn luyện
    best_val_loss = float('inf')
    
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
            
            optimizer.zero_grad()
            predictions = model(rgb_seq, flow_seq)
            
            predictions = predictions.view(-1)
            labels = labels.view(-1)
            predictions = torch.clamp(predictions, min=0.0, max=1.0)

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = (predictions > 0.5).float()
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
                
                predictions = model(rgb_seq, flow_seq)
                predictions = predictions.view(-1)
                labels = labels.view(-1)
                predictions = torch.clamp(predictions, min=0.0, max=1.0)

                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
            
        print(f"\n=> Kết quả Epoch {epoch+1}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("   [+] Đã lưu model tốt nhất (best_model.pth)!\n")

if __name__ == "__main__":
    train_model()