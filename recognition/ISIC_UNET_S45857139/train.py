import torch
import torch.optim as optim
from torch.utils.data import random_split
import torch.nn.functional as F
from model import UNETImproved  # Assuming your model is named like this
from dataset import get_isic_dataloader

# Parameters
num_epochs = 1  # Number of epochs
batch_size = 32  # Batch size
lr = 0.001  # Learning rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Device

# Load Data
root_dir = r'C:\Users\lombo\Desktop\3710_report\ISIC-2017_Training_Data\ISIC-2017_Training_Data'
full_loader = get_isic_dataloader(root_dir, batch_size=batch_size)
train_size = int(0.8 * len(full_loader.dataset))  
val_size = len(full_loader.dataset) - train_size  
train_dataset, val_dataset = random_split(full_loader.dataset, [train_size, val_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = UNETImproved(n_classes=2).to(device)
criterion = torch.nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training Loop
N = 10  # Print every N batches

for epoch in range(num_epochs):
    print(f"\n==== Epoch {epoch} ====")
    
    for phase in ['train', 'val']:  
        if phase == 'train':
            model.train()
            print(f"\n---- Training ----")
            
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                print(f"Batch {i}/{len(train_loader)}", end="\r")
                
        elif phase == 'val':
            model.eval()
            print(f"\n---- Validation ----")
            
            val_loss = 0.0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    print(f"Batch {i}/{len(val_loader)}", end="\r")

            print(f"\n\t Average Validation Loss: {val_loss/len(val_loader)}")
