# predict.py

import torch
from torchvision.utils import save_image
from dataset import test_loader
from modules import VectorQuantizedVAE, generate_sample_from_best_model

INPUT_DIM = 1
DIM = 256
K = 512
DEVICE = torch.device('cuda')

# Reading the best epoch from the file
with open("best_epoch.txt", "r") as file:
    BEST_EPOCH = int(file.readline().strip())

"""
predict(device, latent_dim)
---------------------------
This function loads the best-performing VQ-VAE model, generates an 
image sample using the model, and saves the generated image. The image 
is created based on the best epoch determined during training.

Input: 
- device (str): The device (CPU or GPU) on which to perform computations.
- latent_dim (int): The dimension of the latent space.
Output: None
"""
def predict(device, latent_dim):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device(device if use_cuda else "cpu") # Set the device to GPU if available, otherwise CPU

    # Define the path to the best model checkpoint
    model_path = 'samples6/checkpoint_epoch{}_vqvae.pt'.format(BEST_EPOCH) # using the BEST_EPOCH to load the best model
    
    # Load the best model
    model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)  # Initialize the VQ-VAE model
    model.load_state_dict(torch.load(model_path)) # Load the model's trained weights
    model.eval() # Switch the model to evaluation mode

    # Generate and save a sample image using the best model
    generate_sample_from_best_model(model, test_loader, BEST_EPOCH)
    
if __name__ == "__main__":
    device = "cuda:0"
    latent_dim = 100
    
    predict(device, latent_dim)
