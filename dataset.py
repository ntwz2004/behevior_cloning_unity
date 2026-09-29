import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class AutoDriveDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(data_dir, img_height=200, img_width=240, test_size=0.2, batch_size=32):
    forward_imgs, left_imgs, right_imgs = [], [], []
    
    print("Processing images from:", data_dir)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found!")

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path)
            if img is None: 
                continue
            
            # Resize ภาพ (Width 240, Height 200)
            img = cv2.resize(img, (img_width, img_height))
            
            # ทำภาพต้นฉบับ
            img_orig = img.transpose((2, 0, 1)) / 255.0
            
            # Horizontal Flip (กลับซ้าย-ขวา)
            img_flipped = cv2.flip(img, 1)
            img_flipped = img_flipped.transpose((2, 0, 1)) / 255.0
            
            # เก็บทั้งภาพจริงและภาพ Flip พร้อมสลับ Label ให้สมดุล
            if 'forward' in filename: 
                forward_imgs.append(img_orig)
                forward_imgs.append(img_flipped) # ทางตรง Flip แล้วยังเป็น Forward เหมือนเดิม
            elif 'left' in filename: 
                left_imgs.append(img_orig)
                right_imgs.append(img_flipped)  # เลี้ยวซ้าย Flip แล้วกลายเป็นเลี้ยวขวา
            elif 'right' in filename: 
                right_imgs.append(img_orig)
                left_imgs.append(img_flipped)   # เลี้ยวขวา Flip แล้วกลายเป็นเลี้ยวซ้าย

    print(f"Found (with Flip): Forward={len(forward_imgs)}, Left={len(left_imgs)}, Right={len(right_imgs)}")

    if len(forward_imgs) == 0:
        raise ValueError("No 'forward' images found in dataset!")

    max_samples = len(forward_imgs)
    
    def balance_data(img_list, target_size):
        if len(img_list) == 0:
            print("Warning: Class has 0 images. Skipping balancing for this class.")
            return []
        
        repeated = list(img_list)
        while len(repeated) < target_size:
            diff = target_size - len(repeated)
            repeated.extend(img_list[:diff])
        return repeated[:target_size]

    print(f"Balancing data to {max_samples} samples per class...")
    balanced_forward = forward_imgs
    balanced_left = balance_data(left_imgs, max_samples)
    balanced_right = balance_data(right_imgs, max_samples)

    # รวม Data (0=Forward, 1=Left, 2=Right)
    X = np.array(balanced_forward + balanced_left + balanced_right, dtype=np.float32)
    y = np.array(
        [0] * len(balanced_forward) + 
        [1] * len(balanced_left) + 
        [2] * len(balanced_right), 
        dtype=np.int64
    )

    # แบ่ง Train / Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    train_dataset = AutoDriveDataset(X_train, y_train)
    val_dataset = AutoDriveDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader