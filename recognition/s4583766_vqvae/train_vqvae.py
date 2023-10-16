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
from torch import nn
from torch.nn import functional as F

# Setup file paths
PATH = os.getcwd() + '/'
DATA_PATH_TRAINING_RANGPUR = '/home/groups/comp3710/OASIS'
DATA_PATH_TRAINING_LOCAL = PATH + 'test_img/'
BATCH_SIZE = 32
EPOCHS = 15

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

train_dl, data_variance = load_dataset(data_path_training, BATCH_SIZE)

vqvae = VQVAE(n_hidden_layers=N_HIDDEN_LAYERS, n_residual_hidden_layers=N_RESIDUAL_HIDDENS, n_embeddings=N_EMBEDDINGS, embeddings_dim=EMBEDDINGS_DIM, beta=BETA).to(device)
vqvae = vqvae.to(device)
optimizer = torch.optim.Adam(vqvae.parameters(), lr=LEARNING_RATE)
recon_losses = []
print(vqvae)

vqvae.train()
# Run training
for epoch in range(EPOCHS):
	train_loss = []
	avg_train_loss = 0
	for batch_idx, (data, _) in enumerate(train_dl):
		data = data.to(device)
		optimizer.zero_grad()

		vq_loss, z_q = vqvae(data)
		recons_error = F.mse_loss(z_q, data) / data_variance

		loss = vq_loss + recons_error
		loss.backward()
		optimizer.step()

		train_loss.append(recons_error.item())

		if batch_idx % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
			epoch, batch_idx * len(data), len(train_dl.dataset),
			100. * batch_idx / len(train_dl),
			recons_error.item() / len(data)))

	avg_train_loss = sum(train_loss) / len(train_loss)
	recon_losses.append(avg_train_loss)
	print('====> Epoch: {} Average loss: {:.4f}'.format(
		epoch, avg_train_loss))
	
# print the losses
print("recon loss: ", recon_losses)

# Save the model
torch.save(vqvae.state_dict(), PATH + 'vqvae.pth')

# evaluate the model to determine the loss
vqvae.eval()
test_loss = []
with torch.no_grad():
	for i, (data, _) in enumerate(train_dl):
		data = data.to(device)
		vq_loss, z_q = vqvae(data)
		recons_error = F.mse_loss(z_q, data) / data_variance
		test_loss.append(recons_error.item())

avg_test_loss = sum(test_loss) / len(test_loss)
print('====> Test set loss: {:.4f}'.format(avg_test_loss))




