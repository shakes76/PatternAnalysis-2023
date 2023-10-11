import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import math

class ADNIDataset(Dataset):
    def __init__(self, root_dir, downsample_factor=4, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            downsample_factor (int): Factor by which the image will be downsampled.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.downsample_factor = downsample_factor
        self.image_files = []
        self.labels = []

        for subfolder, label in [('AD', 0), ('NC', 1)]:
            files_in_subfolder = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(root_dir, subfolder)) for f in filenames if f.endswith('.jpeg')]
            self.image_files.extend(files_in_subfolder)
            self.labels.extend([label] * len(files_in_subfolder))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image = Image.open(img_name)

        # Downsample the image
        downsampled_image = image.resize((image.width // self.downsample_factor, image.height // self.downsample_factor), Image.BICUBIC)

        # Apply transforms if any
        if self.transform:
            image = self.transform(image)
            downsampled_image = self.transform(downsampled_image)

        label = self.labels[idx]

        return downsampled_image, image, label

def image_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Assuming Min-Max normalization to [0,1]. Adjust if using Z-score normalization.
    ])

def get_dataloaders(root_dir, batch_size=32):
    # Transforms
    transform = image_transform()

    # Datasets
    train_dataset = ADNIDataset(os.path.join(root_dir, 'train'), transform=transform)
    test_dataset = ADNIDataset(os.path.join(root_dir, 'test'), transform=transform)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader



import torch.nn as nn

class ESPCNN(nn.Module):
    def __init__(self, upscale_factor=4, in_channels=1):
        super(ESPCNN, self).__init__()

        self.first_part = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.sub_pixel = nn.Sequential(
            nn.Conv2d(32, in_channels * (upscale_factor ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor)
        )

    def forward(self, x):
        x = self.first_part(x)
        x = self.sub_pixel(x)
        return x


# ... [rest of the code]
"""
if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 50
    upscale_factor = 4
    batch_size = 32

    # Initialize the model, loss, and optimizer
    model = ESPCNN(upscale_factor=upscale_factor).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load the data
    train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC", batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (downsampled, original, _) in enumerate(train_loader):
            downsampled, original = downsampled.cuda(), original.cuda()

            # Forward pass
            outputs = model(downsampled)
            loss = criterion(outputs, original)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            avg_psnr = 0
            for batch_idx, (downsampled, original, _) in enumerate(test_loader):
                downsampled, original = downsampled.cuda(), original.cuda()
                outputs = model(downsampled)
                mse = criterion(outputs, original)
                psnr = 10 * torch.log10(1 / mse.item())
                avg_psnr += psnr
            avg_psnr = avg_psnr / len(test_loader)
            print(f"Average PSNR on test data: {avg_psnr:.4f} dB")
"""

# ... [rest of the code]

import matplotlib.pyplot as plt

# ... [rest of your code above]

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.001
    num_epochs = 5  # Reduced to 2 for testing
    upscale_factor = 4
    batch_size = 32

    # Initialize the model, loss, and optimizer
    model = ESPCNN(upscale_factor=upscale_factor).cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load the data
    train_loader, test_loader = get_dataloaders("C:\\Users\\soonw\\ADNI\\AD_NC", batch_size=batch_size)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (downsampled, original, _) in enumerate(train_loader):
            downsampled, original = downsampled.cuda(), original.cuda()

            # Forward pass
            outputs = model(downsampled)
            loss = criterion(outputs, original)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (batch_idx+1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Evaluation on test data
        model.eval()
        with torch.no_grad():
            avg_psnr = 0
            for batch_idx, (downsampled, original, _) in enumerate(test_loader):
                downsampled, original = downsampled.cuda(), original.cuda()
                outputs = model(downsampled)
                mse = criterion(outputs, original)
                psnr = 10 * math.log10(1 / mse.item())
                avg_psnr += psnr
            avg_psnr = avg_psnr / len(test_loader)
            print(f"Average PSNR on test data: {avg_psnr:.4f} dB")

    # After training, display an upsampled image and its original counterpart
    model.eval()
    with torch.no_grad():
        for downsampled, original, _ in test_loader:
            downsampled, original = downsampled.cuda(), original.cuda()
            outputs = model(downsampled)
            
            # Convert tensors to numpy arrays for visualization
            downsampled = downsampled.cpu().numpy()[0][0]
            original = original.cpu().numpy()[0][0]
            output = outputs.cpu().numpy()[0][0]
            
            # Display images
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(downsampled, cmap='gray')
            axes[0].set_title('Downsampled Image')
            axes[1].imshow(output, cmap='gray')
            axes[1].set_title('Upsampled by ESPCNN')
            axes[2].imshow(original, cmap='gray')
            axes[2].set_title('Original Image')
            
            for ax in axes:
                ax.axis('off')
            
            plt.show()
            break  # Displaying the first batch's images only

