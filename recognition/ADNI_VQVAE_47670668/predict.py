import torch
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import numpy as np

from dataset import test_loader, train_dataloader
from train import visualize_reconstructions

def get_discrete_latents(model, images):

    with torch.no_grad():
        z_e_x = model._encoder(images)

        z_e_x = model._pre_vq_conv1(z_e_x).permute(0, 2, 3, 1)

        vq_output = model._vq(z_e_x)
        quantize = vq_output["quantize"]
        encoding_indices = vq_output["encoding_indices"].squeeze(-1)

        return quantize, encoding_indices

def show_images(images):
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(15, 5))
    
    # Ensure axs is always a list
    if n == 1:
        axs = [axs]
    
    for i in range(n):
        img = images[i].detach().squeeze().cpu().numpy()  # remove batch dim, and transfer to CPU
        axs[i].imshow(img, cmap='gray')
        axs[i].axis('off')
    plt.show()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load('model_path.pth').to(device)
num_embeddings = 128

# Produce reconstructions for test set
model.eval()  # Set the model to evaluation mode
average_ssim = 0

with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(test_loader):
        inputs = inputs.to(device)

        # Forward pass
        output_dict = model(inputs)

        # Extract the reconstructed images tensor
        reconstructed_images = output_dict['x_recon']

        # Calculate SSIM
        current_ssim = ssim(reconstructed_images, inputs, data_range=1.0)
        average_ssim += current_ssim.item()

        # Visualize the first batch's reconstructions for demonstration
        if batch_idx == 0:
            visualize_reconstructions(inputs, reconstructed_images)

# Average the SSIM over all batches
average_ssim = average_ssim / len(test_loader)

print(f"Average SSIM on test set: {average_ssim:.4f}")  


# 1. Encode the dataset
all_encoding_indices = []
spatial_dim=(56, 56)

for images, _ in train_dataloader:
    _, encoding_indices = get_discrete_latents(model, images.to(device))
    all_encoding_indices.append(encoding_indices.cpu().numpy())

all_encoding_indices = np.concatenate(all_encoding_indices)

# 2. Calculate the histogram for each spatial location
# Reshape the encoding indices to be of shape (num_samples, height, width)
reshaped_indices = np.reshape(all_encoding_indices, (-1, spatial_dim[0], spatial_dim[1]))

# Initialize an array to store the histograms
prob_dist = np.zeros((num_embeddings, spatial_dim[0], spatial_dim[1]))
for i in range(spatial_dim[0]):
    for j in range(spatial_dim[1]):
        # Compute the histogram for the i, j-th spatial location
        hist, _ = np.histogram(reshaped_indices[:, i, j], bins=range(num_embeddings+1))
        # Normalize the histogram to get a probability distribution
        prob_dist[:, i, j] = hist / np.sum(hist)

def generate_samples(model, prob_dist, num_samples=1, spatial_dim=(56, 56)):
    # Sample latent codes from the estimated distribution
    latent_shape = (num_samples,) + spatial_dim
    latent_codes = np.argmax(prob_dist, axis=0)
    latent_codes = np.expand_dims(latent_codes, axis=0)
    latent_codes = np.repeat(latent_codes, num_samples, axis=0)
    latent_codes = torch.tensor(latent_codes, dtype=torch.long).to(device)

    # Convert latent codes to embeddings
    embeddings = model._vq.embeddings(latent_codes).permute(0, 3, 1, 2)

    # Pass embeddings through the decoder
    generated_images = model._decoder(embeddings)

    return generated_images

simple_images = generate_samples(model, prob_dist)


show_images(simple_images)
