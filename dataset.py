import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class AutoDriveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(data_dir, img_width=160, img_height=120, test_size=0.2, batch_size=32):
    forward_imgs, left_imgs, right_imgs = [], [], []
    
    print("Downloading and processing images...")
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            img = cv2.resize(img, (img_width, img_height))
            
            # แปลงมิติจาก OpenCV [H, W, C] เป็น PyTorch [C, H, W]
            img = img.transpose((2, 0, 1)) 
            img = img / 255.0  # Normalize ค่าแสง
            
            if 'forward' in filename: forward_imgs.append(img)
            elif 'left' in filename: left_imgs.append(img)
            elif 'right' in filename: right_imgs.append(img)

    # ทำ Data Balancing เพื่อปรับสัดส่วนข้อมูลให้เท่ากัน
    max_samples = len(forward_imgs)
    def balance_data(img_list, target_size):
        repeated = list(img_list)
        while len(repeated) < target_size:
            diff = target_size - len(repeated)
            repeated.extend(img_list[:diff])
        return repeated

    print(f"Found {max_samples} images for each class. Balancing data...")
    balanced_forward = forward_imgs
    balanced_left = balance_data(left_imgs, max_samples)
    balanced_right = balance_data(right_imgs, max_samples)

    X = np.array(balanced_forward + balanced_left + balanced_right, dtype=np.float32)
    y = np.array([0]*max_samples + [1]*max_samples + [2]*max_samples, dtype=np.int64)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    train_dataset = AutoDriveDataset(X_train, y_train)
    val_dataset = AutoDriveDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader