"""
Created on Sunday Oct 8 2:22:00 2023

This script is for Setting up the code that will be used for training the VQVAE model.

@author: Gabriel Russell
@ID: s4640776

"""

import modules
import torch
import torch.optim as optim
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import dataset
import torch.nn.functional as F
import numpy as np
import os



class TrainVQVAE():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.params = modules.Parameters()
        self.model = modules.VQVAEModel()
        self.model = self.model.to(self.device)
        data = dataset.OASISDataloader()
        self.train_loader = data.get_train()
        self.optimizer = optim.Adam(self.model.parameters(), self.params.learn_rate)
        self.epochs = 2
    
    def train(self):
        reconstruction_error = []
        for epoch in range(self.epochs):
            print(f"Training on Epoch: {epoch + 1}")
            for i in enumerate(self.train_loader):
                batch_num, img = i
                img = img.to(self.device)
                self.optimizer.zero_grad()

                vec_quantizer_loss, recon, _ = self.model(img)
                #Reconstruction loss is mean squared error for images
                reconstruction_err = F.mse_loss(recon, img)/ self.params.data_var
                total_loss = reconstruction_err + vec_quantizer_loss
                total_loss.backward()

                self.optimizer.step()

                reconstruction_error.append(reconstruction_err.item())
                if batch_num % 20 == 0:
                    print('recon_error: %.3f' % np.mean(reconstruction_error[:]))
                    print()
            print(f"EPOCH: {epoch + 1}\n")
            print('Reconstruction Loss: %.3f' % np.mean(reconstruction_error[:]))

        #Filters and Plots Reconstruction Loss values
        train_res_recon_error_smooth = savgol_filter(reconstruction_error, 375, 7)
        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Smoothed NMSE.')
        ax.set_xlabel('iteration')
        plt.savefig("reconstruction_err.png")

        #Saves entire Model after training
        current_dir = os.getcwd()
        model_path = current_dir + "/VQVAE.pth"
        torch.save(self.model, model_path)

