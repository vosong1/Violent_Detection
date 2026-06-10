import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from torch.amp import autocast, GradScaler
from config import Config
from models.valdnet_baseline import TwoStreamBiLSTMModel
from dataset import PreprocessedVideoDataset

def save_training_plot(history, filename="training_history.png"):
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    plt.plot(history["train_loss"], marker='o', label="Train Loss")
    plt.plot(history["val_loss"], marker='o', label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot(history["train_acc"], marker='o', label="Train Acc")
    plt.plot(history["val_acc"], marker='o', label="Val Acc")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(history["train_f1"], marker='o', label="Train F1")
    plt.plot(history["val_f1"], marker='o', label="Val F1")
    plt.title("F1 Score")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def get_model(cfg_class):
    return TwoStreamBiLSTMModel(
        num_classes=1, 
        rnn_hidden_size=cfg_class.LSTM_HIDDEN_SIZE,
        num_rnn_layers=cfg_class.LSTM_LAYERS,
        freeze_backbone=True 
    )

def train_model():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True 

    model = get_model(Config).to(device)

    train_dataset = PreprocessedVideoDataset(Config.PROCESSED_DATA_DIR, phase='train')
    val_dataset = PreprocessedVideoDataset(Config.PROCESSED_DATA_DIR, phase='val')

    # Loss
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=Config.EPOCHS
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    if len(train_loader) == 0 or len(val_loader) == 0:
        print("Dataset rỗng - kiểm tra lại dữ liệu!")
        return

    # AMP
    scaler = GradScaler(device.type, enabled=use_amp)

    # History
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_f1": [],
        "val_f1": []
    }

    best_val_loss = float("inf")
    bad_epochs = 0
    patience = Config.PATIENCE

    normalize_rgb = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    normalize_flow = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(Config.EPOCHS):

        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                m.eval()
                
        model.cnn_rgb.eval()
        model.cnn_op.eval()

        train_loss = 0
        correct = 0
        total = 0

        train_bar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        all_train_preds = []
        all_train_labels = []
        
        optimizer.zero_grad()
        for step, (rgb_seq, flow_seq, labels) in enumerate(train_bar):

            rgb_seq = rgb_seq.to(device)
            flow_seq = flow_seq.to(device)
            labels = labels.to(device)

            B, T, C, H, W = rgb_seq.shape

            if torch.rand(1).item() < 0.3:
                noise = torch.randn_like(rgb_seq) * 0.05
                rgb_seq = torch.clamp(rgb_seq + noise, 0.0, 1.0)
            
            if torch.rand(1).item() < 0.2:
                mask_idx = torch.randint(0, T, (1,)).item()
                rgb_seq[:, mask_idx] = 0.0
                flow_seq[:, mask_idx] = 0.0
                
            smoothed_labels = labels

            rgb_seq = normalize_rgb(rgb_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
            flow_seq = normalize_flow(flow_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)

            with autocast(device.type, enabled=use_amp):
                outputs = model(rgb_seq, flow_seq)
                outputs = outputs.view(-1)
                labels_flat = smoothed_labels.view(-1)
                original_labels = labels.view(-1)
                loss = criterion(outputs, labels_flat)
                loss = loss / Config.GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * Config.GRAD_ACCUM_STEPS

            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_train_preds.extend(preds.detach().cpu().numpy().astype(int))
            all_train_labels.extend(original_labels.detach().cpu().numpy().astype(int))
            
            correct += (preds == original_labels).sum().item()
            total += original_labels.size(0)

            train_bar.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_acc = correct / total
        train_f1 = f1_score(all_train_labels, all_train_preds, zero_division=0)

        # VALIDATION

        model.eval()

        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Val Epoch {epoch+1}")
            all_val_preds = []
            all_val_labels = []
            for rgb_seq, flow_seq, labels in val_bar:

                rgb_seq = rgb_seq.to(device)
                flow_seq = flow_seq.to(device)
                labels = labels.to(device)

                B, T, C, H, W = rgb_seq.shape

                rgb_seq = normalize_rgb(rgb_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)
                flow_seq = normalize_flow(flow_seq.reshape(B*T, C, H, W)).reshape(B, T, C, H, W)

                with autocast(device.type, enabled=use_amp):
                    outputs = model(rgb_seq, flow_seq)
                    outputs = outputs.view(-1)
                    labels_flat = labels.view(-1)
                    loss = criterion(outputs, labels_flat)

                val_loss += loss.item()

                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_val_preds.extend(preds.detach().cpu().numpy().astype(int))
                all_val_labels.extend(labels_flat.detach().cpu().numpy().astype(int))
                
                correct += (preds == labels_flat).sum().item()
                total += labels_flat.size(0)

        val_loss /= len(val_loader)
        val_acc = correct / total
        val_f1 = f1_score(all_val_labels, all_val_preds, zero_division=0)
        
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"\nEpoch {epoch+1}")
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")
        else:
            bad_epochs += 1
            print(f"No improvement: {bad_epochs}/{patience}")

            if bad_epochs >= patience:
                print("Early stopping triggered")
                break

        save_training_plot(history, "training_history.png")

    # LOAD BEST MODEL
    model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
    print("Loaded best model")


if __name__ == "__main__":
    train_model()