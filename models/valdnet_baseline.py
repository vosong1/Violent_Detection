import torch
import torch.nn as nn
import timm

class TwoStreamBiLSTMModel(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, num_rnn_layers=1, freeze_backbone=False):
        super().__init__()
        
        # 1. Khởi tạo 2 mạng Backbone (EfficientNet-B0) như cũ
        self.cnn_rgb = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.cnn_op = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3, num_classes=0)
        
        if freeze_backbone:
            for param in self.cnn_rgb.parameters():
                param.requires_grad = False
            for param in self.cnn_op.parameters():
                param.requires_grad = False

        cnn_feature_dim = 1280
        
        # 2. Lớp Nén (Fusion) Feature: Thay vì cộng, nối RGB và Flow lại và nén qua Linear
        # Dùng LayerNorm thay vì BatchNorm để không bị ảnh hưởng bởi Batch Size nhỏ
        self.feature_fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim * 2, cnn_feature_dim),
            nn.LayerNorm(cnn_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 3. Thay đổi thành BiLSTM bằng cách bật tham số bidirectional=True
        self.rnn = nn.LSTM(input_size=cnn_feature_dim, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=num_rnn_layers, 
                           batch_first=True,
                           bidirectional=True) # <--- BẬT MẠNG 2 CHIỀU TẠI ĐÂY
            
        self.dropout = nn.Dropout(0.5)
        # 4. Lớp Classifications ở cuối sơ đồ
        # VÌ LÀ BILSTM NÊN ĐẦU VÀO PHẢI LÀ: rnn_hidden_size * 2
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x_rgb, x_op):
        B, T, C_rgb, H, W = x_rgb.shape
        _, _, C_op, _, _ = x_op.shape
        
        # --- BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG ---
        x_rgb = x_rgb.view(B * T, C_rgb, H, W) 
        x_op = x_op.view(B * T, C_op, H, W)    
        
        feat_rgb = self.cnn_rgb(x_rgb) 
        feat_op = self.cnn_op(x_op)    
        
        # --- BƯỚC 2: LỚP FUSION (NÉN FEATURE) ---
        # Nối đặc trưng của 2 luồng thay vì cộng (Concat)
        feat_concat = torch.cat((feat_rgb, feat_op), dim=1) # Shape: [B*T, 2560]
        
        # Ép (Nén) qua Linear Layer để mô hình tự học cách kết hợp tốt nhất
        feat_fused = self.feature_fusion(feat_concat)       # Shape: [B*T, 1280]
        
        # --- BƯỚC 3: MẠNG BILSTM ---
        feat_fused = feat_fused.view(B, T, -1) 
        
        # rnn_out shape: [B, T, rnn_hidden_size * 2]
        # hn (hidden state) shape: [num_layers * 2, B, rnn_hidden_size]
        rnn_out, (hn, cn) = self.rnn(feat_fused)
        
        # --- BƯỚC 4: CLASSIFICATIONS (Xử lý đầu ra cho BiLSTM) ---
        # Đối với BiLSTM của PyTorch, hn sẽ chứa các hidden state của cả 2 chiều.
        # Ta lấy trạng thái của layer cuối cùng:
        # hn[-2] là hidden state cuối cùng của hướng XUÔI (Forward)
        # hn[-1] là hidden state cuối cùng của hướng NGƯỢC (Backward)
        
        # Thay vì chỉ lấy Hidden state ở frame cuối (dễ mất thông tin nếu bạo lực xảy ra ở giữa video),
        # Khuyến nghị dùng Average Pooling trên toàn chuỗi thời gian (T frames):
        last_hidden_state = rnn_out.mean(dim=1) # Shape: [Batch, rnn_hidden_size * 2]
        
        last_hidden_state = self.dropout(last_hidden_state)
        
        # Đưa qua lớp Linear để dự đoán kết quả
        output = self.classifier(last_hidden_state) # Shape: [Batch, num_classes]
        
        return output