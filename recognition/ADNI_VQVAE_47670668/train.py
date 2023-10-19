import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from pytorch_msssim import ssim

from dataset import train_dataloader, val_dataloader
from modules import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# Initialize values for incremental variance computation
mean = torch.zeros(1).float().to(device)
M2 = torch.zeros(1).float().to(device)
n = 0

# Loop through the DataLoader and compute incremental variance
for batch in train_dataloader:
    images = batch[0].float().to(device) / 255.0
    batch_mean = images.mean()
    n_batch = images.numel()
    n += n_batch

    delta = batch_mean - mean
    mean += delta * n_batch / n
    M2 += delta * (batch_mean - mean) * n_batch


# Compute variance
if n < 2:
    train_data_variance = float('nan')
else:
    train_data_variance = M2 / (n - 1)


# initialize all variables for training
num_training_updates = 30000
num_epochs = 10

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64

num_embeddings = 512

commitment_cost = 0.25

decay = 0.99

learning_rate = 1e-5


encoder = Encoder(num_hiddens, num_residual_layers, num_residual_hiddens)
decoder = Decoder(num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim)

vq_vae = VectorQuantizer(
      embedding_dim=embedding_dim,
      num_embeddings=num_embeddings,
      commitment_cost=commitment_cost
      )

pre_vq_conv1 = nn.Conv2d(in_channels=num_hiddens, out_channels=embedding_dim,
                         kernel_size=1, stride=1)

model = VQVAEModel(encoder, decoder, vq_vae, pre_vq_conv1,
                   data_variance=train_data_variance)

optimizer = optim.Adam(lr=learning_rate, weight_decay=decay, params=model.parameters())
model.to(device)

torch.autograd.set_detect_anomaly(True)

def train_step(image, label): # Added label as an input, even if you might not use it.
    # Zero the parameter gradients
    optimizer.zero_grad()

    # Move data to device
    image = image.to(device)
    label = label.to(device)

    # Forward pass
    model_output = model(image)
    loss = model_output['loss']

    # Calculate SSIM
    ssim_value = ssim(model_output['x_recon'], image, data_range=1.0)

    # Backward pass and optimization
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()

    return model_output, ssim_value.item()

for epoch in range(num_epochs):  # Added epoch loop

    # Reset training metrics at the start of each epoch
    train_losses = []
    train_recon_errors = []
    train_perplexities = []
    train_vqvae_loss = []
    train_ssim_values = []
    
    for step_index, (image, label) in enumerate(train_dataloader): # Updated data unpacking

        train_results, ssim_value = train_step(image, label)
        train_losses.append(train_results['loss'].item())
        train_ssim_values.append(ssim_value)
        train_recon_errors.append(train_results['recon_error'].item())
        train_perplexities.append(train_results['vq_output']['perplexity'].item())
        train_vqvae_loss.append(train_results['vq_output']['loss'].item())


        if (step_index + 1) % 100 == 0:  # Adjust frequency as needed
            print('Epoch %d/%d - Step %d train loss: %f ' % (epoch + 1, num_epochs, step_index + 1,
                                                              np.mean(train_losses[-100:])) +
                  ('recon_error: %.3f ' % np.mean(train_recon_errors[-100:])) +
                  ('perplexity: %.3f ' % np.mean(train_perplexities[-100:])) +
                  ('vqvae loss: %.3f' % np.mean(train_vqvae_loss[-100:])) +
                  ('ssim: %.3f' % np.mean(train_ssim_values[-100:]))) 

        if step_index == num_training_updates:
            break

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
    
    with torch.no_grad():  # Disable gradient computation during validation
        for image, label in val_dataloader:
            image, label = image.to(device), label.to(device)
            val_results = model(image)
            val_losses.append(val_results['loss'].item())
            val_recon_errors.append(val_results['recon_error'].item())
            val_perplexities.append(val_results['vq_output']['perplexity'].item())
            val_vqvae_loss.append(val_results['vq_output']['loss'].item())

    # Print validation metrics
    print(f"Epoch {epoch + 1}/{num_epochs} - Val loss: {np.mean(val_losses):.3f}, "
          f"recon_error: {np.mean(val_recon_errors):.3f}, "
          f"perplexity: {np.mean(val_perplexities):.3f}, "
          f"vqvae loss: {np.mean(val_vqvae_loss):.3f}")

    model.train()  # Switch back to training mode