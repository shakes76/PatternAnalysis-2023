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
from dataset import load_dataset, show_images
from modules import VQVAE, Decoder, Encoder
from torch import nn
from torch.nn import functional as F
from torchvision.utils import make_grid

# Setup file paths
PATH = os.getcwd() + '/'
DATA_PATH_TRAINING_RANGPUR = '/home/groups/comp3710/OASIS'
DATA_PATH_TRAINING_LOCAL = PATH + 'test_img/'
BATCH_SIZE = 32
EPOCHS = 3

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

def gen_imgs(images, model, device):
	with torch.no_grad():
		images = images.to(device)
		_, x_hat, _ = model(images)
		return x_hat

def main():
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
	og_imgs, _ = next(iter(train_dl))
	grid = make_grid(og_imgs, nrow=8)
	show_images(grid, "before")

	best_loss = float('-inf')

	vqvae.train()
	# Run training
	for epoch in range(EPOCHS):
		train_loss = []
		avg_train_loss = 0
		train_steps = 0
		for data, _ in train_dl:
			data = data.to(device)
			optimizer.zero_grad()

			vq_loss, x_hat, z_q = vqvae(data)
			recons_error = F.mse_loss(x_hat, data)

			loss = vq_loss + recons_error
			loss.backward()
			optimizer.step()

			train_loss.append(recons_error.item())

			gen_img = gen_imgs(data, vqvae, device)
			grid = make_grid(gen_img, nrow=8)
			show_images(grid, epoch)

			if train_steps % 100 == 0:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, train_steps * len(data), len(train_dl.dataset),
				100. * train_steps / len(train_dl),
				recons_error.item() / len(data)))
			train_steps += 1

			# Save the model if the average loss is the best we've seen so far.
			if avg_train_loss > best_loss:
				best_loss = avg_train_loss
				torch.save(vqvae.state_dict(), PATH + 'vqvae.pth')

		# avg_train_loss = sum(train_loss) / len(train_loss)
		# recon_losses.append(avg_train_loss)
		# print('====> Epoch: {} Average loss: {:.4f}'.format(
		# 	epoch, avg_train_loss))

if __name__ == '__main__':
	main()