'''
Example usage of the trained model, generates results and provides visualisations.

Usage: python predict.py <model_path>

Author: Sophie Bates, s4583766.
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
from torchvision.utils import save_image

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
BATCH_SIZE_GAN = 32
EPOCHS = 3
EPOCHS_GAN = 20

# Taken from paper and YouTube video
N_HIDDEN_LAYERS = 64
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3
LR_GAN = 2e-4

def validate_batch(model, dl):
    """Validate the model on a batch of data.

    Can be used for either testing or validation results.
    
    Parameters
    ----------
    model : VQVAE
        The model to validate.
    dl : DataLoader
        The validation DataLoader.

    Returns
    -------
    ssim_list : list
        The SSIM scores for each image in the batch.
    """
    ssim_list = []

    with torch.no_grad():
        for batch in dl:
            # Iterate over each image in the batch and calculate the SSIM score.
            for data in batch:
                sys.stdout.flush()
                data = data.to(device)
                data = data.unsqueeze(0)

                # Run the single image through the model to get the 
                # reconstruction error.
                vq_loss, x_hat, z_q, _ = model(data)
                recons_error = F.mse_loss(x_hat, data)

                # Prepare the images for SSIM calculation.
                img1 = data[0][0].cpu().detach().numpy()
                img2 = x_hat[0][0].cpu().detach().numpy()

                # Calculate SSIM score, and append to list.
                # data_range was important to adjust for continuous values.
                ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
                ssim_list.append(ssim_score.item())
    return ssim_list
       

def evaluate_model(args):
    """Evaluate the model on the test data.
    
    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments.
    """

    # Initialise a VQVAE model
    model = VQVAE(
        n_hidden_layers=N_HIDDEN_LAYERS, 
        n_residual_hidden_layers=N_RESIDUAL_HIDDENS, 
        n_embeddings=N_EMBEDDINGS,
        embeddings_dim=EMBEDDINGS_DIM, 
        beta=BETA
    )	
    # Load the saved weights into the model
    model_path = FILE_SAVE_PATH + args.model_path   
    model.load_state_dict(torch.load(FILE_SAVE_PATH + args.model_path))
    model.to(device)

    # set the device to evaluation mode
    model.eval()

    # get the test data loader
    _, test_dl, _ = get_dataloaders(TRAIN_DATA_PATH, TEST_DATA_PATH, BATCH_SIZE)

    ssim_total = 0
    loss_total = 0
    ssim_list = []
    loss_list = []
    best_recon_before = None 
    best_recon = None
    worst_recon_before = None
    worst_recon = None
    best_ssim = 0
    worst_ssim = 1

    # Iterate over all the test data
    with torch.no_grad():
        for batch in test_dl:
            for data in batch:
                sys.stdout.flush()
                data = data.to(device)
                data = data.unsqueeze(0)

                # Run the single image through the model to get the
                # reconstruction error.
                vq_loss, x_hat, z_q, _ = model(data)
                recons_error = F.mse_loss(x_hat, data)
                loss = recons_error + vq_loss

                # Prepare the images for SSIM calculation.
                img1 = data[0][0].cpu().detach().numpy()
                img2 = x_hat[0][0].cpu().detach().numpy()

                # Calculate SSIM score, and append to list.
                ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
                ssim_total += ssim_score.item()
                ssim_list.append(ssim_score.item())

                # Check whether the SSIM score is the best or worst, 
                # and store the images if so (for plotting later).
                if ssim_score > best_ssim:
                    best_ssim = ssim_score
                    best_recon_before = data
                    best_recon = x_hat
                elif ssim_score < worst_ssim:
                    worst_ssim = ssim_score
                    worst_recon_before = data
                    worst_recon = x_hat

        # Save the best and worst reconstruction images
        save_image(best_recon_before, RUN_IMG_OUTPUT + 'best_recon_before.png')
        save_image(best_recon, RUN_IMG_OUTPUT + 'best_recon.png')
        save_image(worst_recon_before, RUN_IMG_OUTPUT + 'worst_recon_before.png')
        save_image(worst_recon, RUN_IMG_OUTPUT + 'worst_recon.png')

    # Calculate the min, max, and average SSIM scores
    ssim_avg = np.mean(ssim_list)
    print(f'SSIM mean: {ssim_avg:.4f}')
    print(f'Min SSIM score: {min(ssim_list):.4f}')
    print(f'Max SSIM score: {max(ssim_list):.4f}')

    # Calculate the number of images with SSIM >= 0.6 (as per Task 8 specification)
    n_over_threshold = np.sum(np.array(ssim_list) >= 0.6)
    print(f'Number of images with SSIM >= 0.6: {n_over_threshold}, {n_over_threshold/len(ssim_list)*100:.2f}%.')

    # Visualise the SSIM scores for every test image (544 total) as a scatterplot
    plt.scatter(range(len(ssim_list)), ssim_list, label='SSIM')
    plt.legend()
    plt.xlabel('Training image')
    plt.ylabel('SSIM')
    plt.title('SSIM scores for training images')
    # Save the generated plot
    plt.savefig(os.path.join(RUN_IMG_OUTPUT, 'test_ssim_scores.png'))

if __name__ == '__main__':
    # Parse the command line arguments - the path to the trained model
    parser = argparse.ArgumentParser(description='Example usage of the trained model, generates results and provides visualisations.')
    parser.add_argument('model_path', type=str, help='Path to the trained model file')
    args = parser.parse_args()
    evaluate_model(args)