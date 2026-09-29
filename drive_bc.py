import time
import cv2
import torch
import numpy as np
from unity_env import UnityCarEnv
import torch.nn as nn

# 1. โครงสร้างโมเดล (ต้องเหมือนตอนเทรนเป๊ะ)
class NvidiaCNN(nn.Module):
    def __init__(self):
        super(NvidiaCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3), nn.ReLU()
        )
        self.linear_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 18 * 23, 100), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 3) 
        )
    def forward(self, x):
        return self.linear_layers(self.conv_layers(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NvidiaCNN().to(device)

# โหลดไฟล์โมเดลที่เทรนเสร็จแล้ว
model.load_state_dict(torch.load("nvidia_model_unity.pth", map_location=device))
model.eval()

env = UnityCarEnv()
obs, _ = env.reset()
print("========================================")
print("โมเดลเริ่มขับขี่อัตโนมัติ (Classification Mode)")
print("[r] กดเพื่อรีเซ็ตรถกลับจุดสตาร์ต")
print("[q] กดเพื่อหยุดโปรแกรม")
print("========================================")

fixed_throttle = 0.8  

try:
    while True:
        img = cv2.resize(obs, (240, 200))
        img = img.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_tensor = torch.tensor(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_class = outputs.max(1)
            class_idx = predicted_class.item()

        if class_idx == 0:     # Forward
            steering = 0.0
        elif class_idx == 1:   # Left
            steering = -0.8
        elif class_idx == 2:   # Right
            steering = 0.8

        # ส่งคำสั่งให้ Unity วิ่งต่อไปเรื่อยๆ ไม่ว่าจะชนหรือไม่
        obs, reward, done, truncated, _ = env.step([steering, fixed_throttle])

        cv2.imshow("Autonomous Driving", cv2.cvtColor(obs, cv2.COLOR_RGB2BGR))
        
        # รับค่าปุ่มกดผ่านหน้าต่าง OpenCV
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            print("รีเซ็ตรถแบบ Manual...")
            obs, _ = env.reset()
            time.sleep(0.3)
        elif key == ord('q'):
            print("หยุดการทำงาน...")
            break

        # แค่แจ้งเตือนใน Console ว่าเกิดการชน แต่ไม่รีเซ็ต
        if done:
            print("แจ้งเตือน: รถแตะโดนขอบสนาม (OffTrack)", end="\r")

finally:
    env.close()
    cv2.destroyAllWindows()