import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from config import cfg
from models.valdnet_baseline import ValdNetBaseline
from dataset import PreprocessedVideoDataset

def train_model():
    # 1. Khởi tạo Model
    device = torch.device(cfg.DEVICE)
    model = ValdNetBaseline(cfg).to(device)
    
    # 2. Hàm Loss (Binary Cross Entropy cho bài toán phân loại nhị phân: Bạo lực / Bình thường)
    criterion = nn.BCELoss()
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    
    # 4. Load Data
    train_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='train')
    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    
    val_dataset = PreprocessedVideoDataset(data_dir=cfg.PROCESSED_DATA_DIR, phase='val')
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)
    
    # 5. Vòng lặp Huấn luyện (Training Loop)
    best_val_loss = float('inf')
    
    for epoch in range(cfg.EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for batch_idx, (rgb_seq, flow_seq, labels) in enumerate(train_loader):
            rgb_seq = rgb_seq.to(device)
            flow_seq = flow_seq.to(device)
            labels = labels.to(device)
            
            # Xóa gradient cũ
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(rgb_seq, flow_seq)
            
            # Tính Loss
            loss = criterion(predictions, labels)
            
            # Backward pass & Cập nhật trọng số
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Tính Accuracy cho batch
            predicted = (predictions > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
        train_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train
        
        # --- ĐÁNH GIÁ TRÊN TẬP VALIDATION (VAL) ---
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for rgb_seq, flow_seq, labels in val_loader:
                rgb_seq = rgb_seq.to(device)
                flow_seq = flow_seq.to(device)
                labels = labels.to(device)
                
                predictions = model(rgb_seq, flow_seq)
                loss = criterion(predictions, labels)
                
                val_loss += loss.item()
                predicted = (predictions > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
        val_loss /= len(val_loader)
        val_acc = correct_val / total_val
            
        print(f"Epoch [{epoch+1}/{cfg.EPOCHS}] | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
              
        # Lưu model tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("  --> Đã lưu model tốt nhất (best_model.pth)!")

if __name__ == "__main__":
    train_model()   