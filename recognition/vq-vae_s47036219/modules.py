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
    def __init__(self, in_channels, out_channels, intermediate_channels=None):
        super(ResidualBlock, self).__init__()

        if not intermediate_channels:
            intermediate_channels = in_channels // 2

        self._residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(intermediate_channels, out_channels, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._residual_block(x)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64)
        )

    def forward(self, x):
        out = self.layers(x)
        return out


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        
        self.num_embeddings = num_embeddings  # Save as an instance variable
        self.embedding = nn.Embedding(self.num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1./self.num_embeddings, 1./self.num_embeddings)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        x_flat = x.permute(0, 2, 3, 1).contiguous().view(-1, channels)

        # Now x_flat is [batch_size * height * width, channels]
        
        # Calculate distances
        distances = ((x_flat.unsqueeze(1) - self.embedding.weight.unsqueeze(0)) ** 2).sum(-1)
        
        # Find the closest embeddings
        _, indices = distances.min(1)
        encodings = torch.zeros_like(distances).scatter_(1, indices.unsqueeze(1), 1)
        
        # Quantize the input image
        quantized = self.embedding(indices)
        
        # Reshape the quantized tensor to the same shape as the input
        quantized = quantized.view(batch_size, height, width, channels).permute(0, 3, 1, 2)
        
        return quantized

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.layers(x)


class VQVAE(nn.Module):
    def __init__(self, num_embeddings=512, embedding_dim=64):
        super(VQVAE, self).__init__()

        self.encoder = Encoder()
        self.conv1 = nn.Conv2d(64, embedding_dim, kernel_size=1, stride=1)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder()

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.conv1(enc)
        quantized = self.vector_quantizer(enc)
        
        dec = self.decoder(quantized)
        return dec

# Hyperparameters
learning_rate = 1e-3
batch_size = 32
num_epochs = 20
codebook_size = 512

# Weight for L2 and SSIM in final loss
l2_weight = 0
ssim_weight = 1

# Constants for early stopping
patience = 10
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
model = VQVAE(codebook_size).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

# Training Loop
for epoch in range(num_epochs):
    for i, (img, _) in enumerate(train_loader):
        img = img.to(device)  # Move to device

        # Forward pass through the entire model
        recon = model(img)
        
        # Calculate L2 loss
        l2_loss = ((recon - img) ** 2).sum()

        # Calculate SSIM loss
        ssim_loss = 1 - ssim(img, recon)

        # Final Loss
        loss = l2_weight * l2_loss + ssim_weight * ssim_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Assuming you'd still want to update the codebook, though this needs to be integrated with the VQ process
        #model.vector_quantizer.embedding.weight.data = 0.99 * model.vector_quantizer.embedding.weight.data + 0.01 * enc.detach().mean(0).mean(0)

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

    # Update learning rate
    scheduler.step(avg_val_loss)
    # Print current learning rate
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Current Learning Rate: {current_lr}")

    # Early Stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0  # Reset counter when validation loss decreases
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break


# Save Models
torch.save(model.state_dict(), 'vqvae.pth')


