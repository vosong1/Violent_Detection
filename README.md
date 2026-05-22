# Violent_Detection

## Cấu trúc thư mục (Directory Structure)
Dự án được tổ chức theo cấu trúc sau:

```text
Violent_Detection/
│
├── config.py                 # File chứa các tham số cấu hình (Hyperparameters, config model)
├── dataset.py                # Dataset class để nạp dữ liệu .pt vào PyTorch DataLoader
├── process.py                # Script tiền xử lý video (trích xuất RGB & Optical Flow)
├── train.py                  # Script huấn luyện mô hình chính (Training Loop)
├── utils.py                  # Các hàm hỗ trợ (ví dụ: tính Farneback Optical Flow)
│
├── models/
│   └── valdnet_baseline.py   # Định nghĩa kiến trúc mạng ValdNet (mô hình dự đoán bạo lực)
│
├── dataset/
│   └── my_dataset/           # Thư mục chứa dữ liệu gốc
│       ├── train/            # Video dùng để huấn luyện
│       │   ├── violence/     
│       │   └── non_violence/ 
│       ├── val/              # Video dùng để đánh giá (validation)
│       │   ├── violence/     
│       │   └── non_violence/ 
│       └── processed_data/   # Nơi lưu các file Tensor (.pt) sau khi chạy process.py
│
└── README.md
```

## Thứ tự chạy các file (Execution Order)
Để huấn luyện mô hình từ đầu, vui lòng thực hiện tuần tự theo các bước sau( đổi đường dẫn trong cònfig.py nếu cần):

**Bước 1: Chuẩn bị dữ liệu**
* Đặt data vào dataset sau đó chạy file split_data.
**Bước 2: Tiền xử lý dữ liệu (Preprocessing)**
* Chạy lệnh: `python process.py`
* Quá trình này sẽ lấy mẫu khung hình, tính toán Optical Flow và lưu thành các file Tensor (`.pt`) chuẩn bị cho PyTorch trong thư mục `processed_data`.

**Bước 3: Huấn luyện mô hình (Training)**
* Chạy lệnh: `python train.py`
* File này sẽ sử dụng `dataset.py` để load dữ liệu `.pt` lên, bắt đầu huấn luyện và tự động lưu lại file trọng số tốt nhất (`best_model.pth`).