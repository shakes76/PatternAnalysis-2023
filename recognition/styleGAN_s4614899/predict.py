import torch
import matplotlib.pyplot as plt
import os

#import train
import modules

# Random seed to ensure reproducibility
torch.manual_seed(42)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
Z_DIm = 512
W_DIM = 512
IN_CHANNELS = 512
CHANNELS_IMG = 3

gen = modules.Generator(Z_DIm, W_DIM, IN_CHANNELS, CHANNELS_IMG).to(DEVICE)
gen.load_state_dict(torch.load('OASIS_style_gan_generater.pth'))
# eval mode
gen.eval()   

num_samples = 9
z = torch.randn(num_samples, Z_DIm).to(DEVICE)
with torch.no_grad():
    generated_images = gen(z, alpha=1.0, steps=5)  # Assuming reached the 256x256 resolution

# Convert the generated images to a format suitable for visualization
generated_images = (generated_images + 1) / 2  # Convert from [-1, 1] to [0, 1]
generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)  # (batch, height, width, channels)

# Plot
_,ax = plt.subplots(3,3,figsize=(8,8))
plt.suptitle('Generated sample images')

for i in range(3):
    for j in range(3):
        idx = i * 3 + j
        if idx < len(generated_images):
            ax[i][j].imshow(generated_images[idx])

if not os.path.exists("output_images"):
        os.makedirs("output_images")

save_path = os.path.join("output_images", "generated_grid.png")
plt.savefig(save_path)

plt.close()