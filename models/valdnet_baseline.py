import torch
import torch.nn as nn
import timm

class TwoStreamBiLSTMModel(nn.Module):
    def __init__(self, num_classes, rnn_hidden_size=256, num_rnn_layers=1, freeze_backbone=False):
        super().__init__()
        self.cnn_rgb = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.cnn_op = timm.create_model('efficientnet_b0', pretrained=True, in_chans=3, num_classes=0)
        
        if freeze_backbone:
            for param in self.cnn_rgb.parameters():
                param.requires_grad = False
            for param in self.cnn_op.parameters():
                param.requires_grad = False

        cnn_feature_dim = 1280
        
        # 2. Lớp Nén (Fusion)
        self.feature_fusion = nn.Sequential(
            nn.Linear(cnn_feature_dim * 2, cnn_feature_dim),
            nn.LayerNorm(cnn_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        self.rnn = nn.LSTM(input_size=cnn_feature_dim, 
                           hidden_size=rnn_hidden_size, 
                           num_layers=num_rnn_layers, 
                           batch_first=True,
                           bidirectional=True) 
            
        # 3. Lớp Temporal Attention
        # Giúp mô hình tự động tập trung vào các frame quan trọng (có hành động) thay vì cào bằng
        self.attention = nn.Sequential(
            nn.Linear(rnn_hidden_size * 2, rnn_hidden_size),
            nn.Tanh(),
            nn.Linear(rnn_hidden_size, 1),
            nn.Softmax(dim=1)
        )

        self.dropout = nn.Dropout(0.3)
        # 4. Lớp Classifications ở cuối sơ đồ
        self.classifier = nn.Linear(rnn_hidden_size * 2, num_classes)

    def forward(self, x_rgb, x_op):
        B, T, C_rgb, H, W = x_rgb.shape
        _, _, C_op, _, _ = x_op.shape
        
        x_rgb = x_rgb.view(B * T, C_rgb, H, W) 
        x_op = x_op.view(B * T, C_op, H, W)    
        
        feat_rgb = self.cnn_rgb(x_rgb) 
        feat_op = self.cnn_op(x_op)    
        
        #LỚP FUSION (NÉN FEATURE) 
        # Nối đặc trưng của 2 luồng
        feat_concat = torch.cat((feat_rgb, feat_op), dim=1)
        
        feat_fused = self.feature_fusion(feat_concat)       
        
        #MẠNG BILSTM 
        feat_fused = feat_fused.view(B, T, -1) 
        rnn_out, (hn, cn) = self.rnn(feat_fused)
        
        # TEMPORAL ATTENTION
        # Tính trọng số attention cho mỗi frame (B, T, 1)
        attn_weights = self.attention(rnn_out)
        # Nhân trọng số với output của LSTM và tính tổng dọc theo trục thời gian (B, hidden*2)
        context_vector = torch.sum(attn_weights * rnn_out, dim=1)
        
        #CLASSIFICATIONS (Xử lý đầu ra cho BiLSTM) 
        context_vector = self.dropout(context_vector)
        
        output = self.classifier(context_vector)
        
        return output