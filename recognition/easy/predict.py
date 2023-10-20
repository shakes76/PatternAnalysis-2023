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

model = torch.jit.load('recognition/easy/model_scripted.pt')
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 1
transform = transforms.Compose([])
# testing
relative_img_path = 'recognition/easy/testsmall/image/'
current_directory = os.getcwd()

test_image_dir = Path(os.path.join(current_directory, relative_img_path))
output_dir = Path(current_directory)

# Ensure the output directory exists
output_dir.mkdir(parents=True, exist_ok=True)

# Prepare the test dataset and data loader (similar to training)

test_dataset = CustomDataset(image_dir=test_image_dir, mask_dir=None, transform=transform)
print(test_dataset.__len__())
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model.to(device)
# Perform inference and save the results
with torch.no_grad():
    for i, data in enumerate(test_dataloader):
        print(i)
        inputs, _ = data  # only test images, no masks

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