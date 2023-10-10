'''
Training script for the model, including validation, testing, and saving of the model.
Imports the model from modules.py, and the data loader from dataset.py. 
Plots losses and metrics observed during training. 

Sophie Bates, s4583766.
'''

import os

import dataset
import modules
import numpy as np
import torch
import torchvision
from dataset import load_dataset
from modules import VQVAE, Decoder, Encoder

# Setup file paths
PATH = os.getcwd() + '/'
DATA_PATH_TRAINING_RANGPUR = '/home/groups/comp3710/OASIS'
DATA_PATH_TRAINING_LOCAL = PATH + 'test_img/'
BATCH_SIZE = 32
EPOCHS = 4

# Taken from paper and YouTube video
N_HIDDEN_LAYERS = 128
N_RESIDUAL_HIDDENS = 32
N_RESIDUAL_LAYERS = 2

EMBEDDINGS_DIM = 64 # Dimension of each embedding in the codebook (embeddings vector)
N_EMBEDDINGS = 512 # Size of the codebook (number of embeddings)

BETA = 0.25
LEARNING_RATE = 1e-3


# Set the mode to either 'rangpur' or 'local' for testing purposes
mode = 'local'
if mode == 'rangpur':
	data_path_training = DATA_PATH_TRAINING_RANGPUR
elif mode == 'local':
	data_path_training = DATA_PATH_TRAINING_LOCAL

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU...")

# Hyper-parameters

load_dataset(data_path_training, BATCH_SIZE)

vqvae = VQVAE(n_hidden_layers=N_HIDDEN_LAYERS, n_residual_hidden_layers=N_RESIDUAL_HIDDENS, n_embeddings=N_EMBEDDINGS, embeddings_dim=EMBEDDINGS_DIM, beta=BETA).to(device)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)

print(vqvae)
