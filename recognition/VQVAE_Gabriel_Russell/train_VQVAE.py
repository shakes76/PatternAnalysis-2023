"""
Created on Sunday Oct 8 2:22:00 2023

This script is for Setting up the code that will be used for training the VQVAE model.

@author: Gabriel Russell
@ID: s4640776

"""

import modules
import torch
import dataset
import torch.nn.functional as F
import numpy as np


class TrainVQVAE():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = modules.Parameters()
        self.model = modules.VQVAEModel()
        data = dataset.OASISDataloader()
        self.train_loader = data.get_train
        self.optimizer = modules.Optimizer(self.model)
    
    def train(self, reconstruction_error):
        data_var = 0.0338 #Calculated separately
        for i in enumerate(self.train_loader):
            batch_num, img = i
            img = img.to(self.device)
            self.optimizer.zero_grad()

            vec_quantizer_loss, recon, _ = self.model(img)
            #Reconstruction loss is mean squared error for images
            reconstruction_err = F.mse_loss(recon, img)/ data_var
            total_loss = reconstruction_err + vec_quantizer_loss
            total_loss.backward()

            self.optimizer.step()

            reconstruction_error.append(reconstruction_err.item())
            if batch_num % 20 == 0:
                print('recon_error: %.3f' % np.mean(reconstruction_error[-100:]))
                print()

