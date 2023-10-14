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
relative_img_path = 'recognition/easy/datasmall/image/'
relative_mask_path = 'recognition/easy/datasmall/mask/'

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
        self.masks = list(mask_dir.glob("*.png")) if mask_dir else []  # Handle cases with no masks

        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = str(self.images[idx].absolute())

       
        image = read_image(image_path)
        if self.mask_dir:
            
            mask_path = str(self.masks[idx].absolute())
            mask = read_image(mask_path)
        
        # Resize images to the target size
            resize_transform = transforms.Resize(self.target_size)
            image = resize_transform(image)
            mask = resize_transform(mask)

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image, mask
        return image , []

dataset = CustomDataset(image_dir=dir_img, mask_dir=dir_mask, transform=transform)
print(dataset.__len__())
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 1  # Adjust as needed
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader, 0):
        
        print(i)
        inputs, masks = data
        inputs, masks = inputs.to(device), masks.to(device)

        optimizer.zero_grad()
        # print(inputs)
        # Forward pass
        masks = masks.float()
        outputs = model(inputs)
        loss = criterion(outputs, masks)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}")

print("Finished Training")

# testing
relative_img_path = 'recognition/easy/testsmall/image/'


test_image_dir = Path(os.path.join(current_directory, relative_img_path))
output_dir = Path(current_directory)

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare the test dataset and data loader (similar to training)

test_dataset = CustomDataset(image_dir=test_image_dir, mask_dir=None, transform=transform)
print(test_dataset.__len__())
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Perform inference and save the results
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(i)
        inputs, _ = data  # Assuming you only have test images, no masks

        # Move inputs to the device (GPU/CPU) if needed
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)

        # Convert model outputs to images (PIL format) if needed
        output_images = []
        for j in range(outputs.shape[0]):
            output_image = outputs[j].squeeze().cpu().numpy() * 255  # Assuming the output is in the range [0, 1]
            output_image = Image.fromarray(output_image.astype('uint8'))
            output_images.append(output_image)

        # Save the output images
        for j, output_image in enumerate(output_images):
            output_path = output_dir / f'result_{i * batch_size + j}.png'
            output_image.save(output_path)

print("Testing and saving results complete.")