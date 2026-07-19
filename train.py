import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import NvidiaCNN

def train_model():

    DATA_DIR = 'dataset'
    BATCH_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 0.001
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        
    print(f"Using device: {device}")
    
    train_loader, val_loader = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    model = NvidiaCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting training...\n")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        train_acc = 100. * correct / total
        print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}%")
    OUTPUT_NAME = "nvidia_model.pth"
    torch.save(model.state_dict(), OUTPUT_NAME)
    print(f"\nTraining completed successfully! Model file saved to: {OUTPUT_NAME}")

if __name__ == '__main__':
    train_model()