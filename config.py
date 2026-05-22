# config.py

class Config:
    # Đường dẫn (Paths)
    DATA_DIR = "my_dataset"
    PROCESSED_DATA_DIR = "my_dataset/processed_data"

    # Thông số Video
    T_FRAMES = 16              # Số khung hình cho mỗi chuỗi (Sequence Length)
    IMAGE_SIZE = 224           # Kích thước ảnh chuẩn cho EfficientNet
    
    # Thông số Model
    FEATURE_DIM = 1280         # Kích thước vector đầu ra của EfficientNet-B0
    LSTM_HIDDEN_SIZE = 256     # Kích thước hidden state của LSTM
    LSTM_LAYERS = 1            # Số tầng LSTM
    
    # Thông số Huấn luyện (Training)
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    DEVICE = "cuda"            # Đổi thành "cpu" nếu không có GPU

cfg = Config()