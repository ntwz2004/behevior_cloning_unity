import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import NvidiaCNN

def train_model():
    DATA_DIR = 'dataset_5'        # โฟลเดอร์ที่เก็บรูปภาพ
    OUTPUT_NAME = 'nvidia_model_obstecle_5_2.pth' # ชื่อไฟล์เซฟโมเดล
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.0001       
    
    # เลือกอุปกรณ์ประมวลผล
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    # โหลด Dataset (ระบุขนาดภาพ 240x200 ตรงตาม NvidiaCNN)
    train_loader, val_loader = get_data_loaders(
        DATA_DIR, 
        img_width=240, 
        img_height=200, 
        batch_size=BATCH_SIZE
    )
    
    # สร้างโมเดล, Loss function และ Optimizer
    model = NvidiaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = 100. * train_correct / train_total
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}%", end="")
        
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), OUTPUT_NAME)
            print("[Saved]")
        else:
            print()

    print(f"\nTraining completed! Best model saved to: {OUTPUT_NAME}")

if __name__ == '__main__':
    train_model()