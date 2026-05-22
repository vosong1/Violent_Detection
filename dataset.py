# dataset.py
import os
import torch
from torch.utils.data import Dataset

class PreprocessedVideoDataset(Dataset):
    def __init__(self, data_dir, phase='train'):
        """
        data_dir: trỏ vào thư mục "./processed_data"
        phase: 'train' hoặc 'val'
        """
        self.data_dir = os.path.join(data_dir, phase)
        self.file_paths = []
        self.labels = []
        
        # Đọc dữ liệu từ thư mục violence (nhãn = 1)
        v_dir = os.path.join(self.data_dir, 'violence')
        if os.path.exists(v_dir):
            for f in os.listdir(v_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(v_dir, f))
                    self.labels.append(1.0)
        else:
            print(f"Cảnh báo: Không tìm thấy thư mục {v_dir}")
                
        # Đọc dữ liệu từ thư mục non_violence (nhãn = 0)
        nv_dir = os.path.join(self.data_dir, 'non_violence')
        if os.path.exists(nv_dir):
            for f in os.listdir(nv_dir):
                if f.endswith('.pt'):
                    self.file_paths.append(os.path.join(nv_dir, f))
                    self.labels.append(0.0)
        else:
            print(f"Cảnh báo: Không tìm thấy thư mục {nv_dir}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Mở file .pt đã tính sẵn toán học từ Bước 1
        data = torch.load(self.file_paths[idx])
        
        rgb_seq = data['rgb']     # Shape: (16, 3, 224, 224)
        flow_seq = data['flow']   # Shape: (16, 3, 224, 224)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        
        return rgb_seq, flow_seq, label
