'''
Example usage of the trained model, generates results and provides visualisations. 

Sophie Bates, s4583766.
'''

import argparse
import datetime
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dataset import get_dataloaders
from modules import VQVAE, Discriminator, Generator
from skimage.metrics import structural_similarity as ssim
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU...")

# Setup file paths
PATH = os.getcwd() + '/'
FILE_SAVE_PATH = PATH + 'recognition/s4583766_vqvae/gen_img/'
BASE_DATA_PATH = '/home/groups/comp3710/OASIS/'
TRAIN_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_train/'
TEST_DATA_PATH = BASE_DATA_PATH + 'keras_png_slices_test/'

# Create new unique directory for this run
time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_IMG_OUTPUT = FILE_SAVE_PATH + time + "/"
os.makedirs(RUN_IMG_OUTPUT, exist_ok=True)

# Hyper-parameters
BATCH_SIZE = 32
BATCH_SIZE_GAN = 256
EPOCHS = 3
EPOCHS_GAN = 20

# Taken from paper and YouTube video
N_HIDDEN_LAYERS = 128
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3
LR_GAN = 2e-4

def evaluate_model(args):
    # load the trained model
    model = VQVAE(
        n_hidden_layers=N_HIDDEN_LAYERS, 
        n_residual_hidden_layers=N_RESIDUAL_HIDDENS, 
        n_embeddings=N_EMBEDDINGS,
        embeddings_dim=EMBEDDINGS_DIM, 
        beta=BETA
    )	
    model.load_state_dict(torch.load(args.model_path))

    # set the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # get the test data loader
    _, test_loader = get_dataloaders()

    # evaluate the SSIM results of the model
    ssim_total = 0
    mse_total = 0
    loss_total = 0
    ssim_list = []
    mse_list = []
    loss_list = []
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon, _, _ = model(data)
            ssim = F.mse_loss(data, recon, reduction='mean')
            mse = F.mse_loss(data, recon, reduction='mean')
            loss = ssim + mse
            ssim_total += ssim.item()
            mse_total += mse.item()
            loss_total += loss.item()
            ssim_list.append(ssim.item())
            mse_list.append(mse.item())
            loss_list.append(loss.item())

            # save the input and output images
            save_image(data, f'input_{i}.png')
            save_image(recon, f'output_{i}.png')

    # calculate the average SSIM score
    ssim_avg = ssim_total / len(test_loader)
    mse_avg = mse_total / len(test_loader)
    loss_avg = loss_total / len(test_loader)
    print(f'Average SSIM score: {ssim_avg:.4f}')
    print(f'Average MSE score: {mse_avg:.4f}')
    print(f'Average loss score: {loss_avg:.4f}')

    # plot the SSIM, MSE, and loss curves
    plt.plot(ssim_list, label='SSIM')
    plt.plot(mse_list, label='MSE')
    plt.plot(loss_list, label='Loss')
    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.title('SSIM, MSE, and Loss Curves')
    plt.savefig(os.path.join('RUN_IMG_OUTPUT', 'ssim_mse_loss.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example usage of the trained model, generates results and provides visualisations.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()
    evaluate_model(args)