# config.py

class Config:
    # Đường dẫn (Paths)
    DATA_DIR = "my_dataset"
    PROCESSED_DATA_DIR = "my_dataset/processed_data"

    # Thông số Video
    T_FRAMES = 16              # Tăng lên 16 frame để LSTM có góc nhìn dài và chi tiết hơn về chuyển động
    IMAGE_SIZE = 224           # Kích thước chuẩn của EfficientNet-B0 để trích xuất đặc trưng tốt nhất
    
    # Thông số Model
    FEATURE_DIM = 1280         # Kích thước vector đầu ra của EfficientNet-B0
    LSTM_HIDDEN_SIZE = 64      # Giảm từ 256 xuống 64 để chống Overfitting
    LSTM_LAYERS = 1            # Số tầng LSTM
    
    # Thông số Huấn luyện (Training)
    BATCH_SIZE = 8             # Buộc phải GIẢM MẠNH (xuống 4 hoặc 6) để nhường VRAM cho lượng Frames lớn hơn
    GRAD_ACCUM_STEPS = 4       # Tích lũy gradient để mô phỏng Batch Size lớn (vd: 8 x 4 = 32)
    LEARNING_RATE = 1e-4       # Giảm LR xuống 1e-4 để tránh dao động (Bounce) làm Loss dội ngược
    WEIGHT_DECAY = 1e-4        # Giảm L2 xuống để mô hình không bị "kìm hãm" quá mức, dễ dàng vượt ngưỡng 0.7
    PATIENCE = 5               # Số Epoch chờ trước khi Early Stopping
    MIN_DELTA = 0.001          # Độ giảm loss tối thiểu để được coi là mô hình có cải thiện
    EPOCHS = 50
    DEVICE = "cuda"            # Đổi thành "cpu" nếu không có GPU

cfg = Config()