import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path
from unet.modules import UNet
import os
from torchvision import transforms
from PIL import Image
import numpy as np
from dataset import CustomDataset

# define dice coefficient as lost function for cnn
def dice_coefficient(predicted, target, smooth=1):
    intersection = (predicted * target).sum()
    union = predicted.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1 - dice

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predicted, target):
        loss = dice_coefficient(predicted, target, self.smooth)
        return loss

# Define U-Net model
model = UNet(n_class=1)  # Adjust the number of classes according to your task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss() # Use appropriate loss for your task

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define data loading and augmentation
transform = transforms.Compose([])
batch_size = 10

current_directory = os.getcwd()
relative_img_path = 'recognition/easy/datasmall/image/'
relative_mask_path = 'recognition/easy/datasmall/mask/'

dir_img = Path(os.path.join(current_directory, relative_img_path))
dir_mask = Path(os.path.join(current_directory, relative_mask_path))
# print(dir_img.absolute())
# print(dir_mask.absolute())
# print(list(dir_mask.glob("*.png")))

# Create custom dataset for your data

dataset = CustomDataset(image_dir=dir_img, mask_dir=dir_mask, transform=transform)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


# Training loop
num_epochs = 10  # Adjust as needed

model = model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        
        # print(i)
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)
        # print(inputs.shape)
        # print( inputs.max())
        # print(inputs.min())
        # print(inputs.unique())
        # print(masks.shape)
        # print(masks.unique())

        # print("loss", dice_coefficient(inputs, masks))
        optimizer.zero_grad()

        # Forward pass
        masks = masks.float()
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # print(running_loss,loss.item())

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}")

print("Finished Training")

model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('recognition/easy/model_scripted.pt') # Save

