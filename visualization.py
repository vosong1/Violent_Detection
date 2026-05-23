import torch
import matplotlib.pyplot as plt

# 1. Load 1 file tensor bất kỳ
data = torch.load("D:\\Violent_Detection\\my_dataset\\processed_data\\train\\violence\\fi90_xvid.pt")
rgb_seq = data['rgb']    # Shape: (16, 3, 224, 224)
flow_seq = data['flow']  # Shape: (16, 3, 224, 224)

# 2. Chọn khung hình thứ 8 (ở giữa video) để xem
frame_idx = 15


# 3. Chuyển đổi ngược: (C, H, W) -> (H, W, C) để matplotlib vẽ được
img_rgb = rgb_seq[frame_idx].permute(1, 2, 0).numpy()
img_flow = flow_seq[frame_idx].permute(1, 2, 0).numpy()

# 4. Vẽ lên màn hình
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img_rgb)
ax[0].set_title("Luồng RGB (Đã chuẩn hóa)")
ax[1].imshow(img_flow)
ax[1].set_title("Luồng Optical Flow")
plt.show()