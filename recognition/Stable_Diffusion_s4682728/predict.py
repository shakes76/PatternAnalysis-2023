from imports import *
from dataset import *
from modules import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def beta_schedule(timesteps):
    start = 0.0001
    end = 0.02
    return torch.linspace(start, end, timesteps)

# Initialize models and dataset
num_steps = 1000
betas = beta_schedule(num_steps)

# Load trained model
model_path = "diffusion_network99.pth"
diffusion_process = DiffusionProcess(betas, num_steps).to(device)
diffusion_network = DiffusionNetwork()
diffusion_network.load_state_dict(torch.load(model_path))
diffusion_network.to(device)
diffusion_network.eval()  # set to evaluation mode

# Load a sample batch of data
batch_size = 3
dataloader = process_dataset(batch_size=batch_size, is_validation=True)
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
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

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

# Save images
save_dir = os.path.expanduser("~/demo_eiji/sd/images")
full_path = os.path.join(save_dir, "image_visualization.png")
plt.savefig(full_path)