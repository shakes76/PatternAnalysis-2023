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
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter




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
for epochs in range(5):
    for i in enumerate(train_data):
        batch_num, img = i
        img = img.to(device)
        optimizer.zero_grad()

        vec_quantizer_loss, recon, _ = model(img)
        #Reconstruction loss is mean squared error for images
        reconstruction_err = F.mse_loss(recon, img)/ data_var
        total_loss = reconstruction_err + vec_quantizer_loss
        total_loss.backward()

        optimizer.step()

        reconstruction_err_vals.append(reconstruction_err.item())
        if batch_num % 20 == 0:
            print('recon_error: %.3f' % np.mean(reconstruction_err_vals[-100:]))
            print()
    print("Epoch + " + str(epochs))
train_res_recon_error_smooth = savgol_filter(reconstruction_err_vals, 375, 7)


f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Smoothed NMSE.')
ax.set_xlabel('iteration')

plt.savefig("reconstruction_err.png")