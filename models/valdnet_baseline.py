import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class ValdNetBaseline(nn.Module):
    def __init__(self, cfg):
        super(ValdNetBaseline, self).__init__()
        
        # 1. Khởi tạo bộ trích xuất đặc trưng (EfficientNet-B0)
        # Bỏ đi lớp Classification cuối cùng, chỉ lấy Feature Extractor
        cnn = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(
            cnn.features,
            cnn.avgpool,
            nn.Flatten() # Đầu ra sẽ là vector 1280 chiều
        )
        
        # 2. Mạng phân tích chuỗi thời gian (Bi-LSTM)
        # Vì là Bi-LSTM (2 chiều) nên đầu ra thực tế sẽ là: HIDDEN_SIZE * 2
        self.bi_lstm = nn.LSTM(
            input_size=cfg.FEATURE_DIM,
            hidden_size=cfg.LSTM_HIDDEN_SIZE,
            num_layers=cfg.LSTM_LAYERS,
            batch_first=True,    # Định dạng Input: (Batch, Seq, Features)
            bidirectional=True
        )
        
        # Thêm Dropout trước khi vào LSTM để phạt thêm
        self.dropout_feature = nn.Dropout(0.5)
        
        # 3. Lớp phân loại cuối cùng (Ra quyết định bạo lực hay không)
        # Đầu vào là 2 * LSTM_HIDDEN_SIZE (do chạy 2 chiều tiến & lùi)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.LSTM_HIDDEN_SIZE * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.7),     # Tăng Dropout từ 0.5 lên 0.7 để giảm Overfitting mạnh hơn
            nn.Linear(128, 1)    # Bỏ Sigmoid theo chuẩn BCEWithLogitsLoss
        )

    def forward(self, rgb_seq, flow_seq):
        # Kích thước đầu vào dự kiến: (Batch, T, C, H, W)
        batch_size, T, C, H, W = rgb_seq.size()
        
        # Gộp Batch và T lại để cho vào EfficientNet cùng lúc (tối ưu tốc độ)
        rgb_reshaped = rgb_seq.reshape(batch_size * T, C, H, W)
        flow_reshaped = flow_seq.reshape(batch_size * T, C, H, W)
        
        # Trích xuất đặc trưng
        # Kết quả: (Batch * T, 1280)
        feat_rgb = self.feature_extractor(rgb_reshaped)
        feat_flow = self.feature_extractor(flow_reshaped)
        
        # --- TẦNG FUSION (GIAI ĐOẠN 1: CỘNG TRỰC TIẾP) ---
        feat_fused = feat_rgb + feat_flow 
        
        # Tách lại ra định dạng chuỗi: (Batch, T, 1280)
        feat_fused = feat_fused.reshape(batch_size, T, -1)
        
        # Regularization: Áp dụng Dropout cho feature hỗn hợp
        feat_fused = self.dropout_feature(feat_fused)
        
        # Cho qua mạng Bi-LSTM
        # out: (Batch, T, Hidden_size * 2)
        lstm_out, (h_n, c_n) = self.bi_lstm(feat_fused)
        
        # Lấy hidden state cuối cùng của cả 2 chiều tiến (forward) và lùi (backward)
        # h_n có kích thước (num_layers * 2, batch, hidden_size)
        # Chúng ta lấy 2 layer trên cùng: tiến (h_n[-2]) và lùi (h_n[-1])
        last_time_step_out = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        
        # Phân loại và lấy xác suất
        out_prob = self.classifier(last_time_step_out)
        
        return out_prob