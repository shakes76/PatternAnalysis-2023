# --------------------------------------------------------------------------------
# File: train.py
# Author: Indira Devi Rusvandy
# Date: 2023-10-20
# Description:
#   This script is dedicated to training a Vector Quantized Variational Autoencoder (VQ-VAE) 
#   using PyTorch. It includes the setup of the VQ-VAE model, its loss function, and the 
#   training loop. The script also incorporates evaluation metrics such as Structural Similarity 
#   Index Measure (SSIM) and performs KMeans initialization on the encoder outputs.
#
#   The script assumes the availability of 'dataset.py' for data loading and 'modules.py' for
#   model components (like Encoder, Decoder, etc.). It also visualizes reconstructions from the
#   VQ-VAE during training and validation phases.
#
# Usage:
#   Run this script to train the VQ-VAE model. Adjust the hyperparameters and model configuration 
#   as needed for your specific dataset and training requirements. The script outputs training and 
#   validation losses, SSIM values, and reconstructed images for visual inspection.
#
#   Example:
#       python train.py
# --------------------------------------------------------------------------------

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim
from sklearn.cluster import KMeans

from dataset import train_dataloader, val_dataloader
from modules import *


def visualize_reconstructions(original_images, reconstructed_images, num_samples=10):
    # This function assumes the images are tensors with shape [batch_size, channels, height, width]
    
    num_samples = min(num_samples, original_images.size(0))  # Ensure num_samples is within bounds
    
    _, axs = plt.subplots(2, num_samples, figsize=(15, 5))
    for i in range(num_samples):
        axs[0, i].imshow(original_images[i].permute(1, 2, 0).cpu().numpy())
        axs[1, i].imshow(reconstructed_images[i].detach().permute(1, 2, 0).cpu().numpy())
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.tight_layout()
    plt.show()

def vq_vae_loss(inputs, reconstructions, quantized_latents, latents, beta=0.25):
    # This function computes the loss from the VQVAE model for training
    # Reconstruction loss
    recon_loss = F.mse_loss(reconstructions, inputs)

    # VQ-VAE vector quantization loss
    vq_loss = F.mse_loss(quantized_latents.detach(), latents)

    # Commitment loss
    commit_loss = F.mse_loss(quantized_latents, latents.detach())

    # Formula given from paper
    total_loss = recon_loss + vq_loss + beta * commit_loss
    return total_loss, recon_loss, vq_loss, commit_loss


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize all variables for training
num_training_updates = 10000
num_epochs = 10

num_hiddens = 256
num_residual_hiddens = 64
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 128

commitment_cost = 0.25

decay = 1e-5

learning_rate = 1e-5

encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

vq = VectorQuantizer(embedding_dim, num_embeddings, commitment_cost)

with torch.no_grad():
    for batch, _ in train_dataloader:  # Iterate over DataLoader to get a batch of data
        batch = batch.to(device)  
        latents = encoder(batch)  # Obtain latents for this batch of data
        latents = latents.permute(0, 2, 3, 1).reshape(-1, embedding_dim)
        break  # Break after the first batch.
    
    kmeans = KMeans(n_clusters=num_embeddings)
    kmeans.fit(latents.cpu().numpy())

vq.embeddings.weight.data.copy_(torch.from_numpy(kmeans.cluster_centers_))

pre_vq_conv1 = nn.Conv2d(num_hiddens, embedding_dim, kernel_size=1, stride=1)

model = VQVAEModel(encoder, decoder, vq, pre_vq_conv1).to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

torch.autograd.set_detect_anomaly(True)

def train_step(image, label, beta):
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Move data to device
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    model_output = model(image)

    # Extract necessary outputs
    reconstructions = model_output['x_recon']
    quantized_latents = model_output['vq_output']['quantize']
    latents = model_output['z']

    # Call the VQ-VAE loss function with the appropriate beta value
    loss, recon_loss, vq_loss, commit_loss = vq_vae_loss(image, reconstructions, quantized_latents, latents, beta=beta)

    # Calculate SSIM
    ssim_value = ssim(reconstructions, image, data_range=1.0)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    return model_output, loss, recon_loss, vq_loss, ssim_value.item()


for epoch in range(num_epochs):  # Added epoch loop

    # Reset training metrics at the start of each epoch
    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []
    train_ssim_values = []

    for step_index, (image, label) in enumerate(train_dataloader): # Updated data unpacking

        train_results, loss, recon_loss, vq_loss, ssim_value = train_step(image, label, commitment_cost)
        train_losses.append(loss.detach().cpu())
        train_ssim_values.append(ssim_value)
        train_recon_errors.append(recon_loss.detach().cpu())
        train_perplexities.append(train_results['vq_output']['perplexity'].item())
        train_vqvae_loss.append(vq_loss.detach().cpu())


        if (step_index + 1) % 100 == 0:  # Adjust frequency as needed
            print('Epoch %d/%d - Step %d train loss: %f ' % (epoch + 1, num_epochs, step_index + 1,
                                                              np.mean(train_losses[-100:])) +
                  ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                  ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                  ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])) +
                  ('ssim: %.3f' % np.mean(train_ssim_values[-100:])))

        if step_index == num_training_updates:
            break

    scheduler.step()

    # Visualization logic
    with torch.no_grad():
        reconstructed_images = train_results['x_recon']
        reconstructed_images = torch.clamp(reconstructed_images, 0, 1)
    visualize_reconstructions(image, reconstructed_images)

    # After training loop, begin validation
    model.eval()  # Switch to evaluation mode

    # Initialize validation metrics

    val_losses = []
    val_recon_errors = []
    val_perplexities = []
    val_vqvae_loss = []
    val_ssim_values = []

    with torch.no_grad():  # Disable gradient computation during validation
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)
            val_results = model(image)

            reconstructions = val_results['x_recon']
            quantized_latents = val_results['vq_output']['quantize']
            latents = val_results['z']

            loss, recon_loss, vq_loss, commit_loss = vq_vae_loss(image, 
                                                                 reconstructions, 
                                                                 quantized_latents, 
                                                                 latents, 
                                                                 beta=commitment_cost)
            ssim_value = ssim(reconstructions, image, data_range=1.0)

            val_losses.append(loss.detach().cpu())
            val_recon_errors.append(recon_loss.detach().cpu())
            val_perplexities.append(val_results['vq_output']['perplexity'].item())
            val_vqvae_loss.append(vq_loss.detach().cpu())
            val_ssim_values.append(ssim_value.detach().cpu())


    # Print validation metrics
    print(f"Epoch {epoch + 1}/{num_epochs} - Val loss: {np.mean(val_losses):.3f}, "
          f"recon_error: {np.mean(val_recon_errors):.3f}, "
          f"perplexity: {np.mean(val_perplexities):.3f}, "
          f"vqvae loss: {np.mean(val_vqvae_loss):.3f}, "
          f"ssim: {np.mean(val_ssim_values):.3f}")

    model.train()  # Switch back to training mode


