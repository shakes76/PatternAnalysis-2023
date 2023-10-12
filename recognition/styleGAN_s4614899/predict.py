import torch
import matplotlib.pyplot as plt

#import train
import modules


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
    generated_images = gen(z, alpha=1.0, steps=5)  # Assuming you've reached the 256x256 resolution, adjust the step accordingly

# Convert the generated images to a format suitable for visualization
generated_images = (generated_images + 1) / 2  # Convert from [-1, 1] to [0, 1]
generated_images = generated_images.cpu().numpy().transpose(0, 2, 3, 1)  # (batch, height, width, channels)

# Plot
fig, axarr = plt.subplots(1, num_samples, figsize=(15,15))
for idx, img in enumerate(generated_images):
    axarr[idx].imshow(img)
    axarr[idx].axis('off')
#plt.show()

save_path = "generated_grid.png"
plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 


plt.close()
