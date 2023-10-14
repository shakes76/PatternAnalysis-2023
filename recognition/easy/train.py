import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from pathlib import Path
from unet.UNet_model import UNet
import os
from torchvision import transforms
from PIL import Image

# Define your U-Net model
model = UNet(n_class=1)  # Adjust the number of classes according to your task
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define your loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Use appropriate loss for your task
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define data loading and augmentation
transform = transforms.Compose([])
batch_size = 4

# Assuming you have your data organized in a directory structure like this:
# dir_img contains input images
# dir_mask contains corresponding masks
current_directory = os.getcwd()
relative_img_path = 'recognition/easy/data/image/'
relative_mask_path = 'recognition/easy/data/mask/'

dir_img = Path(os.path.join(current_directory, relative_img_path))
dir_mask = Path(os.path.join(current_directory, relative_mask_path))
# print(dir_img.absolute())
# print(dir_mask.absolute())
# print(list(dir_mask.glob("*.png")))

# Create custom dataset for your data
from torchvision.io import read_image

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(image_dir.glob("*.jpg"))
        self.masks = list(mask_dir.glob("*.png"))

        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = str(self.images[idx].absolute())
        mask_path = str(self.masks[idx].absolute())
        image = read_image(image_path)
        mask = read_image(mask_path)
        
        # Resize images to the target size
        resize_transform = transforms.Resize(self.target_size)
        image = resize_transform(image)
        mask = resize_transform(mask)

        

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

dataset = CustomDataset(image_dir=dir_img, mask_dir=dir_mask, transform=transform)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model.forward(inputs)
        loss = criterion(outputs, masks)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}")

print("Finished Training")

# Validation can be added here to monitor model performance during training
