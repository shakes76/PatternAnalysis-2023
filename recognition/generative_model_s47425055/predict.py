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

def predict(device, latent_dim):
    # Check if CUDA is available
    use_cuda = torch.cuda.is_available()
    device = torch.device(device if use_cuda else "cpu")

    # Load the best model
    model_path = 'samples6/checkpoint_epoch{}_vqvae.pt'.format(BEST_EPOCH) # using the BEST_EPOCH to load the best model
    model = VectorQuantizedVAE(INPUT_DIM, DIM, K).to(DEVICE)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate and save the image
    generate_sample_from_best_model(model, test_loader, BEST_EPOCH)
    

if __name__ == "__main__":
    device = "cuda:0"
    latent_dim = 100
    
    predict(device, latent_dim)
