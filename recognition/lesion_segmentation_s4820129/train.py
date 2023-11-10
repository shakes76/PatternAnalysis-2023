from modules import ImprovedUNET
from dataset import ISICdataset
from torch.utils.data import DataLoader
import wandb
import torch
from utilities import accuracy, DiceLoss
from torch import nn
from torch.utils.data import random_split
wandb.init(project="project_name", name="current_run")


# Initialize Dataset and Dataloader
image_dir = '/home/groups/comp3710/ISIC2018/ISIC2018_Task1-2_Training_Input_x2'
truth_dir = '/home/groups/comp3710//ISIC2018/ISIC2018_Task1_Training_GroundTruth_x2'
fulldataset = ISICdataset(image_dir, truth_dir)
n_elements = 2594
#splitting into training and validation sets
train_dataset, validation_dataset = random_split(fulldataset, [round(n_elements*0.8),n_elements- round(n_elements*0.8)])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
# Initialize Model, Loss and Optimizer
n_channels = 3 # RGB Images
n_classes = 1  # Background and Object
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ImprovedUNET(n_channels, n_classes)
model = model.to(device)
criterion = DiceLoss() # DiceLoss()
lr_init = 0.001
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr_init, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_init * (0.985 ** epoch))

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    
    epoch_loss = 0
    epoch_acc = 0
    epoch_dice = 0

    for i, (image, truth) in enumerate(train_dataloader):
        model.train()
        image = image.to(device)
        truth = truth.to(device).float()
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(image)
        
        # Calculate loss
        loss = criterion(pred, truth)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

    for i, (image, truth) in enumerate(validation_dataloader):
        model.eval()
        image = image.to(device)
        truth = truth.to(device)
        pred = model(image)

        with torch.no_grad():
            loss = criterion(pred,truth)

            acc = accuracy(pred, truth)

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_dice += 1-loss.item()
            
            wandb.log({'Batch loss': loss.item(), 'Batch accuracy': acc, 'Batch dice_score': 1-loss.item()})
        
    scheduler.step()
    torch.save(model.state_dict(), f'checkpoints/checkpoint{epoch}.pth')
    wandb.log({'Epoch loss': epoch_loss/len(train_dataloader), 'Epoch accuracy': epoch_acc/len(train_dataloader), 'Epoch dice score': epoch_dice/len(train_dataloader)}) 
    print(f'Epoch {epoch+1}, Avg Loss {epoch_loss/len(train_dataloader)}')

torch.save(model.state_dict(), 'checkpoints/final.pth')


