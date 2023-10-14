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
        self.val_loader = data.get_validate()
        self.test_loader = data.get_test()
        self.optimizer = optim.Adam(self.model.parameters(), self.params.learn_rate)
        self.epochs = 2
        self.reconstruction_err = []
        self.validation_err = []
    
    def train(self):
        for epoch in range(self.epochs):
            print(f"VQVAE Training on Epoch: {epoch + 1}")
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

                self.reconstruction_err.append(reconstruction_err.item())
                # if batch_num % 20 == 0:
                #     print('VQVAE Reconstruciton error for training: %.3f' % np.mean(self.reconstruction_err[:]))
                #     print()
            print(f"EPOCH: {epoch + 1}\n")
            print('Reconstruction Loss: %.3f' % np.mean(self.reconstruction_err[-302:]))
        
        #Filters and Plots Reconstruction Loss values
        train_res_recon_error_smooth = savgol_filter(self.reconstruction_err, 604, 7)
        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Reconstruction Loss after training')
        ax.set_xlabel('iteration')
        plt.savefig("Output_files/reconstruction_err_train.png")

        #Saves entire Model after training
        current_dir = os.getcwd()
        model_path = current_dir + "/Models/VQVAE.pth"
        torch.save(self.model, model_path)

    def validate(self):
        model = torch.load("Models/VQVAE.pth")
        self.model.eval()
        with torch.no_grad():
            for i in enumerate(self.val_loader):
                batch_num, img = i
                img = img.to(self.device)

                vec_quantizer_loss, recon, _ = model(img)
                reconstruction_err = F.mse_loss(recon, img)/ self.params.data_var

                self.validation_err.append(reconstruction_err.item())
                # if batch_num % 20 == 0:
                #     print('Reconstruction Error for validation set: %.3f' % np.mean(self.validation_err[:]))
                #     print()
            #Filters and Plots Reconstruction Loss values
        train_res_recon_error_smooth = savgol_filter( self.validation_err, 35, 7)
        f = plt.figure(figsize=(16,8))
        ax = f.add_subplot(1,2,1)
        ax.plot(train_res_recon_error_smooth)
        ax.set_yscale('log')
        ax.set_title('Reconstruction Loss after Validating')
        ax.set_xlabel('iteration')
        plt.savefig("Output_files/reconstruction_err_validate.png")

    def test_reconstructions(self):
        test_input = next(iter(self.test_loader))
        test_input = test_input.to(self.device)
        encoded = self.model.encoder(test_input)
        conv = self.model.conv_layer(encoded)
        _,quantized_results,_,embeddings = self.model.quantizer(conv)
        reconstructions = self.model.decoder(quantized_results)
        #print(f"reconstrcuted image batch shape is {reconstructions.shape}")

        batch = reconstructions.shape[0]
        num_rows = 4  # Number of rows in the grid
        num_cols = batch // num_rows  # Number of columns in the grid
        for i in range(reconstructions.shape[0]):
            plt.subplot(num_rows, num_cols, i + 1)
            image = reconstructions[i][0].detach().cpu()
            plt.imshow(image, cmap='gray')  # Assuming grayscale images
            plt.axis('off')

        plt.tight_layout(pad = 0, w_pad = 0, h_pad = 0)
        plt.savefig("Output_files/VQVAE_reconstructed_images.png")




