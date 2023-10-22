"""
Example usage of the trained model
"""
import argparse
import torch
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from modules import Generator, MappingNetwork
from config import device, log_resolution, output_path
from config import modelName, z_dim, w_dim, save_path
from train import trainStyleGAN2

log_resolution = 8
z_dim = 512
w_dim = 512

parser = argparse.ArgumentParser()
parser.add_argument('-load_path_mapping', default=f'./Models/MAPPING_NETWORK_{modelName}.pth', help='Provide the load path of the mapping network')
parser.add_argument('-load_path_generator', default=f'./Models/GENERATOR_{modelName}.pth', help='Provide the load path of the generator network')
parser.add_argument('-num_output', default=64, help='Number of generated outputs')
parser.add_argument('-plt_title', default="Generated Images", help='Provide a title for the plot')
parser.add_argument('-train_models', default="FALSE", help='Train a new series of models (using the parameters defined in config.pg) and then perform inference. If specified, the provided model paths will be ignored.')
args = parser.parse_args()

# Check if training is needed
if args.train_models != 'FALSE':
    # Train the model
    trainStyleGAN2()

# Create the mapping network
mapping_network = MappingNetwork(z_dim, w_dim).to(device)
# If we trained a model, then load in that specific model
if args.train_models != 'FALSE':
    mapping_network.load_state_dict(torch.load(save_path + f"MAPPING_NETWORK_{modelName}.pth"
, map_location=device))
else:
    mapping_network.load_state_dict(torch.load(args.load_path_mapping, map_location=device))

# TODO: refactor these into a different modules so it can be used in both the predict & train files
def get_w(batch_size, log_resolution):
    '''
    Creates a style latent vector w, from a random noise z latent vector.
    '''
    # Random noise z latent vector
    z = torch.randn(batch_size, w_dim).to(device)

    # Forward pass z through the mapping network to generate w latent vector
    w = mapping_network(z)
    return w[None, :, :].expand(log_resolution, -1, -1)

def get_noise(batch_size):
    '''
    Generates a random noise vector for a batch of images
    '''
    noise = []
    resolution = 4

    for i in range(log_resolution):
        if i == 0:
            n1 = None
        else:
            n1 = torch.randn(batch_size, 1, resolution, resolution, device=device)
        n2 = torch.randn(batch_size, 1, resolution, resolution, device=device)

        noise.append((n1, n2))

        resolution *= 2

    return noise

# Create the generator network.
generator = Generator(log_resolution, w_dim).to(device)
if args.train_models != 'FALSE':
    generator.load_state_dict(torch.load(save_path + f"GENERATOR_{modelName}.pth", map_location=device))
else:
    generator.load_state_dict(torch.load(args.load_path_generator, map_location=device))

# Load the trained generator weights.
print(generator)

print(f'Number of images to output: {args.num_output}')

# Turn off gradient calculation to speed up the process.
with torch.no_grad():
    # Get generated images from the noise vector using the trained generator.
    # Get latent vector style vecotr (w).
    w = get_w(args.num_output, log_resolution)

    # Get some random noise
    noise = get_noise(args.num_output)
    generated_img = generator(w, noise).detach().cpu()

# Display the generated image.
plt.figure(figsize=(4, 4))
plt.axis("off")
plt.margins(x=0)
plt.title(args.plt_title)
plt.imshow(np.transpose(vutils.make_grid(generated_img, padding=2, normalize=True), (1, 2, 0)))
plt.savefig(f'{save_path}/generatedImages.png', bbox_inches='tight', dpi=600)