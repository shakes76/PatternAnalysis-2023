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
import numpy as np
import torch


#Initialise GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Retrive datasets -> train, validate and test dataloaders
load_data = OASISDataloader()
train_data = load_data.get_train()
validate_data = load_data.get_validate()
test_data = load_data.get_test()

#Load VQVAE model and optimizer 
model = VQVAEModel()
model = model.to(device)
load_optim = Optimizer()
optimizer = load_optim.Adam
p = Parameters()

#Calculated separately
data_var = 2.1689048e-06


model.train()
reconstruction_err_vals = []
for i in enumerate(train_data):
    batch_num, img = i
    img = img.to(device)
    optimizer.zero_grad()

    vec_quantizer_loss, recon, _ = model(img)
    #Reconstruction loss is mean squared error for images
    reconstruction_err = F.mse_loss(recon, img)/ data_var
    total_loss = reconstruction_err + vec_quantizer_loss
    total_loss.backwards()

    optimizer.step()

    reconstruction_err_vals.append(reconstruction_err.item())
    if (i+1) % 100 == 0:
        print('%d iterations' % (i+1))
        print('recon_error: %.3f' % np.mean(reconstruction_err_vals[-100:]))
        print()
