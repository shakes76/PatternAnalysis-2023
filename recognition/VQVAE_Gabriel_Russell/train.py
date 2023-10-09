"""
Created on Monday Sep 18 12:20:00 2023

This script is for training, validating, testing and saving the VQVAE model.
The model is imported from modules.py and the data loader is imported
from dataset.py. Appropriate metrics are plotted during training.

@author: Gabriel Russell
@ID: s4640776

"""
from modules import *
from dataset import *
from train_VQVAE import *
from train_DCGAN import *
import numpy as np
import torch
import matplotlib.pyplot as plt

#Load Parameters
params = Parameters()

#Intialise VQVAE training
VQVAE_init = TrainVQVAE()

#Train VQVAE model, save it to current dir and save plot of reconstruction losses
VQVAE_init.train()

#Load VQVAE model to be used
VQVAE_model = torch.load("VQVAE.pth")

#Load VQVAE encodings into DCGAN
Gan_dataset = DCGAN_Dataset(VQVAE_model)
Gan_loader = DataLoader(Gan_dataset, batch_size = params.Gan_batch_size)
# a = next(iter(Gan_loader))
# print(a.shape)

#Initialise DCGAN training
DCGAN_init = TrainDCGAN(Gan_loader)
#Train DCGAN model, save it to current dir
DCGAN_init.train()


