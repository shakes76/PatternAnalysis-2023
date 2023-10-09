from modules import DiffusionProcess, DiffusionNetwork
from dataset import process_dataset
import torch
import matplotlib.pyplot as plt

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load trained model
model_path = "diffusion_network.pth"
diffusion_network = DiffusionNetwork().to(device)
diffusion_network.load_state_dict(torch.load(model_path))
diffusion_network.eval()  # Set the model to evaluation mode

# Initialize DiffusionProcess
betas = [0.1] * 100  # Example list of betas for 100 steps
num_steps = 100
diffusion_process = DiffusionProcess(betas, num_steps).to(device)

# Load a sample batch of data
batch_size = 3
dataloader = process_dataset(batch_size=batch_size)
sample_batch = next(iter(dataloader)).to(device)

# Apply diffusion process
diffused_sample = diffusion_process(sample_batch)

# Generate images from diffused images
with torch.no_grad():
    output_samples = diffusion_network(diffused_sample)

# Move data to CPU for visualization
sample_batch = sample_batch.cpu()
diffused_sample = diffused_sample.cpu()
output_samples = output_samples.cpu()

# Visualize
fig, axes = plt.subplots(10, 3, figsize=(15, 15))

for i in range(3):
    axes[i, 0].imshow(sample_batch[i][0], cmap='gray')
    axes[i, 0].set_title('Original Image')
    
    axes[i, 1].imshow(diffused_sample[i][0], cmap='gray')
    axes[i, 1].set_title('Diffused Image')
    
    axes[i, 2].imshow(output_samples[i][0], cmap='gray')
    axes[i, 2].set_title('Generated Image')

for ax_row in axes:
    for ax in ax_row:
        ax.axis('off')

plt.show()
