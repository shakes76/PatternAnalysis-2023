"""
Hugo Burton
s4698512
20/09/2023

predict.py
implements usage of *trained* model.
Prints out results and provides visualisations
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
from dataset import OASIS
# Replace with the actual module where your model is defined
from modules import VectorQuantizedVAE
from train import generate_samples

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters. These can be arbitrary as it's actually read from the save file
num_channels = 1
image_x, image_y = 256, 256
# Size/dimensions within latent space
hidden_dim = 32     # Number of neurons in each layer
K = 32      # Size of the codebook

# Define model.
model = VectorQuantizedVAE(input_channels=num_channels,
                           output_channels=num_channels, hidden_channels=hidden_dim, num_embeddings=K)
# Load the saved model
model.load_state_dict(torch.load(os.path.join(".", "ModelParams", "best.pt"))
                      )  # Update the path accordingly
# Move model to CUDA GPU
model = model.to(device)
# Set the model to evaluation mode
model.eval()

oasis_data_path = os.path.join(".", "datasets", "OASIS")


# Define a transformation to apply to the images (e.g., resizing)
transform = transforms.Compose([
    transforms.Resize((image_x, image_y)),  # Adjust the size as needed
    transforms.Grayscale(num_output_channels=num_channels),
    transforms.ToTensor(),
])

oasis = OASIS(oasis_data_path, transform=transform)

# Define a data loader for generating samples
sample_loader = oasis.test_loader

# Get a batch of data for generating samples
sample_images, _ = next(iter(sample_loader))

# Generate samples
with torch.no_grad():
    # Replace your_fixed_images with actual data
    fake_images = generate_samples(sample_images, model, device)

# Create the "OASIS_generated" directory if it doesn't exist
output_dir = './OASIS_generated'
os.makedirs(output_dir, exist_ok=True)

# Generate and save each sample
for i in range(len(fake_images)):
    sample = fake_images[i].cpu().numpy().squeeze()
    sample_filename = os.path.join(output_dir, f'fake_{i + 1:03d}.png')
    plt.imsave(sample_filename, sample, cmap='gray')


# Visualize and save the generated samples
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i].cpu().numpy().squeeze(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'samples.png'))  # Save the figure

# Display the generated samples
plt.show()
