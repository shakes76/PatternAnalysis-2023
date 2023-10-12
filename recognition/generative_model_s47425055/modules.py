# modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
import numpy as np
import matplotlib.pyplot as plt
import torchvision


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


# function to compute SSIM
def compute_ssim(x, x_tilde):
    # Ensure that the tensors are detached and moved to the CPU
    x_np = x.cpu().detach().numpy()
    x_tilde_np = x_tilde.cpu().detach().numpy()
    
    # Get batch size
    batch_size = x_np.shape[0]
    
    # Initialize a list to store SSIM values for each image in the batch
    ssim_values = []

    # Calculate SSIM for each image in the batch
    for i in range(batch_size):
        ssim_val = ssim(x_np[i, 0], x_tilde_np[i, 0], data_range=1)  # Assuming the images are (batch, channel, height, width), and channel=1 for grayscale
        ssim_values.append(ssim_val)
    
    # Calculate mean SSIM for the batch
    mean_ssim = np.mean(ssim_values)
    return mean_ssim




def plot_losses_and_scores():
    # Extract training losses for reconstruction and VQ
    train_recons_losses, train_vq_losses = zip(*train_losses_epoch)
    
    # Extract validation losses
    val_recons_losses, val_vq_losses = zip(*val_losses)
    
    epochs = range(len(train_recons_losses))

    # Plotting
    plt.figure(figsize=(12, 5))

    # Plot reconstruction losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_recons_losses, '-o', label='Training Recon Loss')
    plt.plot(epochs, val_recons_losses, '-o', label='Validation Recon Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot VQ losses
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_vq_losses, '-o', label='Training VQ Loss')
    plt.plot(epochs, val_vq_losses, '-o', label='Validation VQ Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot SSIM scores
    plt.subplot(1, 3, 3)
    plt.plot(epochs, ssim_scores, '-o', label='Validation SSIM')
    plt.xlabel("Epochs")
    plt.ylabel("SSIM Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig('samples2/loss_ssim_plot.png', bbox_inches='tight')
    plt.close()
def save_and_display_images(images, filename, nrow=8):
    """Saves and displays a grid of images."""
    # Save the image
    torchvision.utils.save_image(images, filename, nrow=nrow)

    # Display the image
    grid_img = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'{filename}_plot.png', bbox_inches='tight')
    plt.close()

def generate_samples(model, test_loader, epoch):
    """Generates and saves reconstructed samples for a given epoch."""
    model.eval()  # Set model to evaluation mode
    x, _ = next(iter(test_loader))  # Get a batch of samples
    x = x[:32].to(DEVICE)

    # Reconstruct the images using the model
    x_tilde, _, _ = model(x)
    images = (torch.cat([x, x_tilde], 0).cpu().data + 1) / 2

    # Save and display the reconstructed images
    filename = f'samples3/vqvae_reconstructions_{epoch}'
    save_and_display_images(images, filename, nrow=8)

def generate_sample_from_best_model(model, test_loader, best_epoch):
    """Generates and saves a sample using the best model from a given epoch."""
    # Load the best model's weights
    model.load_state_dict(torch.load(f'samples3/checkpoint_epoch{best_epoch}_vqvae.pt'))
    model.eval()

    # Get a sample from the test set
    x, _ = next(iter(test_loader))
    x = x[:32].to(DEVICE)

    # Reconstruct the image using the model
    x_tilde, _, _ = model(x)

    # Save the reconstructed image
    filename = 'samples3/best_model_sample'
    save_and_display_images(x_tilde.cpu().data, filename)


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # z_e_x - (B, D, H, W)
        # emb   - (K, D)

        emb = self.embedding.weight
        z_e_x_reshaped = z_e_x.permute(0, 2, 3, 1).contiguous().view(-1, z_e_x.shape[1])  # (B*H*W, D)
        emb_reshaped = emb  # since emb is already (K, D)

        # Calculate distances between reshaped tensors
        z_e_x_norm = (z_e_x_reshaped**2).sum(1, keepdim=True)  # (B*H*W, 1)
        emb_norm = (emb_reshaped**2).sum(1, keepdim=True).t()  # (1, K)

        dists = z_e_x_norm + emb_norm - 2 * torch.mm(z_e_x_reshaped, emb_reshaped.t())  # (B*H*W, K)

        dists = dists.view(z_e_x.shape[0], z_e_x.shape[2], z_e_x.shape[3], -1)  # reshape back to (B, H, W, K)
        latents = dists.min(-1)[1]
        return latents


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents, z_e_x

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x
