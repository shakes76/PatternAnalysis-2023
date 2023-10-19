import torch
import matplotlib.pyplot as plt
import os

# Import the necessary modules
import modules

# Set a random seed for reproducibility
torch.manual_seed(42)

# Determine the device to use (mps if available, else CPU)
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
Z_DIM = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3

# Initialize the generator
gen = modules.Generator(Z_DIM, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)

# Load the pre-trained generator model weights
gen.load_state_dict(torch.load('Generator.pth'))

# Set the generator to evaluation mode
gen.eval()

# Generate a specified number of sample images
num_samples = 9
z = torch.randn(num_samples, Z_DIM).to(DEVICE)
with torch.no_grad():
    generated_images = gen(z, alpha=1.0, steps=5)

# Transform and prepare the generated images for visualization
generated_images = (generated_images + 1) / 2
generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)

# Create a 3x3 grid to display the generated images
_, ax = plt.subplots(3, 3, figsize=(8, 8))
plt.suptitle('Generated images')

# Display the generated images in the grid
for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        if idx < len(generated_images):
            ax[i][j].imshow(generated_images[idx])

# Ensure the output_images directory exists
if not os.path.exists("output_images"):
    os.makedirs("output_images")

# Save the generated image grid to a file
save_path = os.path.join("output_images", "generated_images.png")
plt.savefig(save_path)

# Close the plot
plt.close()
