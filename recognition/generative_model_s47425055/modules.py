# modules.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision.utils import save_image
from itertools import cycle
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import time
from pathlib import Path
from torchvision import datasets, transforms
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

DEVICE = torch.device('cuda')


"""
to_scalar(arr)
-------------
This function  converts a PyTorch tensor or a list of tensors into scalars. 
If arr is a list, it iterates through the list and extracts the scalar 
values of each tensor element using the .item() method. If arr is not a 
list, it directly extracts the scalar value.

Input: array (tensor)
Output: scalar or list of scalars

"""
def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()

"""
weights_init(m)
--------------
This function is used for initializing the weights of convolutional layers 
in a neural network module m. It checks if the module m is a convolutional 
layer (by searching for the string 'Conv' in its class name) and then 
initializes the weights using Xavier uniform initialization 
(nn.init.xavier_uniform_) and sets biases to zero.

Input: m (PyTorch module)
Output: None
"""
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

"""
compute_ssim(x, x_tilde)
-----------------------
 This function calculates the Structural Similarity Index (SSIM) between 
 two batches of images represented by PyTorch tensors x and x_tilde. 
 It first converts these tensors to NumPy arrays, computes SSIM values 
 for individual images in the batch, and then returns the mean SSIM 
 score for the entire batch. SSIM measures the structural similarity 
 between two images, with higher values indicating greater similarity.

 Input: x (PyTorch Tensor), x_tilde (PyTorch Tensor)
 Output: Mean SSIM score (float)
"""
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

"""
plot_losses_and_scores(train_losses_epoch, val_losses, ssim_scores)
--------------------------------------------------------------------
This function creates a plot that displays the training and validation 
losses for both reconstruction and VQ aspects of a VQ-VAE model, as well 
as the SSIM scores over different epochs. It helps visualize the training 
progress of the model.

Input: 
 - train_losses_epoch (list of tuples): Training reconstruction and 
    VQ losses for each epoch.
 - val_losses (list of tuples): Validation reconstruction and VQ losses 
    for each epoch.
 - ssim_scores (list of floats): SSIM scores for each epoch.
 Output: None
"""
def plot_losses_and_scores(train_losses_epoch, val_losses, ssim_scores):
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
    plt.savefig('models6/loss_ssim_plot.png', bbox_inches='tight')
    plt.close()

"""
generate_samples(model, test_loader, epoch)
-------------------------------------------
This function generates and saves reconstructed image samples using the 
VQ-VAE model for a specific training epoch. It extracts a batch of input 
images from the test loader, passes them through the model for 
reconstruction, and saves the original and reconstructed images as a grid 
in an image file.

Inputs:
- model (PyTorch Model): The VQ-VAE model.
- test_loader (PyTorch DataLoader): DataLoader for the test dataset.
- epoch (int): Current training epoch.
Output: None
"""
def generate_samples(model, test_loader, epoch):
    """Generates and saves reconstructed samples for a given epoch."""
    model.eval()  # Set model to evaluation mode
    test_loader_iter = cycle(test_loader) # Initialize cycling iterator here
    x, _ = next(test_loader_iter)  # Get a batch of samples using the cycling iterator
    x = x[:32].to(DEVICE)

    # Reconstruct the images using the model
    x_tilde, _, _ = model(x)
    images = (torch.cat([x, x_tilde], 0).cpu().data + 1) / 2

    # Save and display the reconstructed images
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'samples6/vqvae_reconstructions_{epoch}.png', bbox_inches='tight')
    plt.close()

"""
generate_sample_from_best_model(model, test_loader, best_epoch)
---------------------------------------------------------------
This function generates and saves a single image sample using the 
best-trained VQ-VAE model. It loads the best model's weights, extracts a 
batch of input images from the test loader, passes them through the model 
for reconstruction, computes the SSIM score between the original and 
reconstructed images, and saves the original and reconstructed images with 
the SSIM score as part of the title.

Input:
- model (PyTorch Model): The best-trained VQ-VAE model from a specific epoch
- test_loader (PyTorch DataLoader): DataLoader for the test dataset.
- best_epoch (int): The epoch where the best model was achieved.
Output: None
"""
def generate_sample_from_best_model(model, test_loader, best_epoch):
    """Generates and saves a sample using the best model from a given epoch."""
    model.eval()
    # Load the best model's weights
    test_loader_iter = cycle(test_loader) # Initialize cycling iterator here
    # Get a sample from the test set
    x, _ = next(test_loader_iter)  # Get a batch of samples using the cycling iterator
    x = x[:32].to(DEVICE)

    # Reconstruct the image using the model
    x_tilde, _, _ = model(x)
    images = (torch.cat([x, x_tilde], 0).cpu().data + 1) / 2

    # Compute SSIM
    ssim_val = compute_ssim(x, x_tilde)

    # Save the reconstructed image
    grid_img = torchvision.utils.make_grid(images, nrow=8)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))

    # Add SSIM score to the title
    plt.title(f"SSIM: {ssim_val:.4f}")

    plt.savefig(f'models6/best_model_sample.png', bbox_inches='tight')
    plt.close()

# Define a class for the VQ embedding module
class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        # Create an embedding layer with K embeddings and dimension D
        self.embedding = nn.Embedding(K, D)
        # Initialize the embedding weights with uniform random values
        self.embedding.weight.data.uniform_(-1./K, 1./K)

    def forward(self, z_e_x):
        # Reshape the input tensor and the embedding weights
        emb = self.embedding.weight
        z_e_x_reshaped = z_e_x.permute(0, 2, 3, 1).contiguous().view(-1, z_e_x.shape[1])  # (B*H*W, D)
        emb_reshaped = emb  # since emb is already (K, D)

        # Calculate distances between reshaped tensors
        z_e_x_norm = (z_e_x_reshaped**2).sum(1, keepdim=True)  # (B*H*W, 1)
        emb_norm = (emb_reshaped**2).sum(1, keepdim=True).t()  # (1, K)
        dists = z_e_x_norm + emb_norm - 2 * torch.mm(z_e_x_reshaped, emb_reshaped.t())  # (B*H*W, K)

        # Reshape dists back to (B, H, W, K) and find the indices of the minimum values along the last dimension
        dists = dists.view(z_e_x.shape[0], z_e_x.shape[2], z_e_x.shape[3], -1)  # reshape back to (B, H, W, K)
        latents = dists.min(-1)[1]
        return latents

# Define a class for the residual block
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
        # Apply a series of convolutional and normalization layers, then add the input tensor
        return x + self.block(x)

# Define a class for the Vector Quantized Variational Autoencoder (VQ-VAE)
class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            # Define the encoder network
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        # Create the codebook as an instance of the VQEmbedding class
        self.codebook = VQEmbedding(K, dim)

        # Define the decoder network
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

        # Initialize network weights using the weights_init function
        self.apply(weights_init)

    def encode(self, x):
        # Forward pass through the encoder network
        z_e_x = self.encoder(x)
        # Encode the latents using the codebook
        latents = self.codebook(z_e_x)
        return latents, z_e_x

    def decode(self, latents):
        # Get the embeddings from the codebook and reshape them
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        # Forward pass through the decoder network
        x_tilde = self.decoder(z_q_x)
        return x_tilde, z_q_x

    def forward(self, x):
        # Encode the input data and decode the latents to get the reconstruction
        latents, z_e_x = self.encode(x)
        x_tilde, z_q_x = self.decode(latents)
        return x_tilde, z_e_x, z_q_x
