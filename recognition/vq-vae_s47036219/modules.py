import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pytorch_ssim


def ssim(img1, img2, C1=0.01**2, C2=0.03**2):
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    
    sigma1_sq = (img1 - mu1).pow(2).mean(dim=[2, 3], keepdim=True)
    sigma2_sq = (img2 - mu2).pow(2).mean(dim=[2, 3], keepdim=True)
    sigma12 = ((img1 - mu1)*(img2 - mu2)).mean(dim=[2, 3], keepdim=True)
    
    ssim_n = (2*mu1*mu2 + C1) * (2*sigma12 + C2)
    ssim_d = (mu1.pow(2) + mu2.pow(2) + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_val = ssim_n / ssim_d

    return ssim_val.mean()

# Check for CUDA availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on: ", device)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv(x)

# VQ-VAE Components
# VQ-VAE Components
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv2d(256, 512, 4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = self.conv(x)
        #print(f"Encoder output shape: {x.shape}")
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.deconv(x)
        #print(f"Decoder output shape: {x.shape}")
        return x

class VectorQuantizer(nn.Module):
    def __init__(self, codebook_size=512, code_dim=512):
        super(VectorQuantizer, self).__init__()
        self.codebook = nn.Parameter(torch.randn(codebook_size, code_dim).cuda())
        
    def forward(self, z):
        # z shape: [batch, num_vectors_per_batch, dim]
        # codebook shape: [num_codebook_entries, dim]
        
        # Calculate distances
        dist = ((z.unsqueeze(2) - self.codebook.unsqueeze(0).unsqueeze(0)) ** 2).sum(-1)
        
        # Find the closest codebook entry for each z vector
        _, indices = dist.min(-1)
        
        # Fetch the corresponding codebook vectors
        # The shape is the same as z ([batch, num_vectors_per_batch, dim])
        z_q = self.codebook[indices, :]

        return z_q

# Hyperparameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 75
codebook_size = 512

# Weight for L2 and SSIM in final loss
l2_weight = 0
ssim_weight = 1

# Constants for early stopping
patience = 75
best_val_loss = float('inf')
counter = 0


# Data Loaders
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5], std=[0.5])  
])

train_dataset = datasets.ImageFolder(root='C:/Users/Connor/Documents/comp3710/dataset/ADNI/AD_NC/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Validation DataLoader
val_dataset = datasets.ImageFolder(root='C:/Users/Connor/Documents/comp3710/dataset/ADNI/AD_NC/test', transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


# Models and Optimizers
encoder = Encoder().to(device)  # Move to device
decoder = Decoder().to(device)  # Move to device
vector_quantizer = VectorQuantizer().to(device)  # Move to device

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, verbose=True)

# Training Loop
for epoch in range(num_epochs):
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)  # Move to device

        # Forward pass
        z = encoder(img)
        batch_size, _, H, W = z.shape

        z = z.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W, -1)
        z_q = vector_quantizer(z)
        z_q = z_q.view(batch_size, H, W, 512).permute(0, 3, 1, 2).contiguous()

        # Decoder
        recon = decoder(z_q)

        # Calculate L2 loss
        l2_loss = ((recon - img) ** 2).sum()

        # Calculate SSIM loss
        ssim_loss = 1 - ssim(img, recon)

        # Final Loss
        loss = l2_weight * l2_loss + ssim_weight * ssim_loss

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # Update codebook
        vector_quantizer.codebook.data = 0.99 * vector_quantizer.codebook.data + 0.01 * z.detach().mean(0).mean(0)

    # Validation phase
    val_losses = []
    encoder.eval()
    decoder.eval()
    vector_quantizer.eval()
    with torch.no_grad():
        for i, (img, _) in enumerate(val_loader):
            img = img.to(device)
            
            # Validation forward pass
            z = encoder(img)
            batch_size, _, H, W = z.shape

            z = z.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W, -1)
            z_q = vector_quantizer(z)
            z_q = z_q.view(batch_size, H, W, 512).permute(0, 3, 1, 2).contiguous()

            # Decoder
            recon = decoder(z_q)

            # Validation losses
            l2_loss = ((recon - img) ** 2).sum()
            ssim_loss = 1 - ssim(img, recon)
            loss = l2_weight * l2_loss + ssim_weight * ssim_loss

            val_losses.append(loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {loss.item():.4f}, Validation Loss: {avg_val_loss:.4f}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Save Models
torch.save(encoder.state_dict(), 'encoder.pth')
torch.save(decoder.state_dict(), 'decoder.pth')
torch.save(vector_quantizer.state_dict(), 'vectorquantizer.pth')


from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2  # Import OpenCV
import torch

# Assuming your existing PyTorch models are already loaded
# encoder, decoder, vector_quantizer, and codebook

encoder.eval()  # Set the encoder model to evaluation mode
decoder.eval()  # Set the decoder model to evaluation mode
vector_quantizer.eval()  # Set the vector quantizer to evaluation mode

with torch.no_grad():  # Turn off gradients for the upcoming operations
    for img, _ in train_loader:  # Loop through batches in the data loader
        img = img.cuda()  # Move the images to the GPU
        z = encoder(img)  # Encode the images to latent space

        batch_size, _, H, W = z.shape
        z = z.permute(0, 2, 3, 1).contiguous().view(batch_size, H * W, -1)
        
        z_q = vector_quantizer(z)
        z_q = z_q.view(batch_size, H, W, 512).permute(0, 3, 1, 2).contiguous()
        
        recon = decoder(z_q)  # Decode the quantized vectors

        # Calculate SSIM
        original_img = img.cpu().numpy().squeeze(1)  # Convert to numpy and remove color channel dimension
        reconstructed_img = recon.cpu().numpy().squeeze(1)  # Convert to numpy and remove color channel dimension
        
        ssim_val = ssim(original_img[0], reconstructed_img[0], data_range=reconstructed_img.max() - reconstructed_img.min())  # Calculate SSIM

        print(f'SSIM: {ssim_val}')  # Output SSIM value

        # Output images
        plt.subplot(1, 2, 1)
        plt.title('Original')
        plt.imshow(original_img[0], cmap='gray')  # Show original image
        
        plt.subplot(1, 2, 2)
        plt.title('Reconstructed')
        plt.imshow(reconstructed_img[0], cmap='gray')  # Show reconstructed image

        plt.show()  # Show the plot

        break  # Exit the loop after one iteration for demonstration
