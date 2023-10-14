##################################
#
# Author: Joshua Wallace
# SID: 45809978
#
###################################

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset
import matplotlib.pyplot as plt
from utils import PRINT_AT, DEVICE

class TrainVQVAE() :
    
    def __init__(self, model: nn.Module, dataset: Dataset, lr=1e-3, wd=0, epochs=10, savepath='./models/vqvae') :

        self.savepath = savepath
        self.lr = lr
        self.wd = wd
        self.epochs =epochs

        self.model = model.to(DEVICE)
        self.dataset = dataset
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)        
        self.losses = list()

    def train(self) -> None :
        if self.dataset.train_unloaded() :
            self.dataset.load_train()
        
        self.model.train()
        start = time.time()
        for epoch in range(self.epochs) :
            epoch_loss = list()
            for i, (data, label) in enumerate(self.dataset.get_train()) :
                data = data.to(DEVICE)
                self.optimiser.zero_grad()

                embedding_loss, x_hat, perplexity = self.model(data)
                recon_loss = torch.mean((x_hat - data)**2) / self.dataset.train_var()
                loss = recon_loss + embedding_loss

                loss.backward()
                self.optimiser.step()

                epoch_loss.append(loss.item())

                if i % PRINT_AT == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {i+1}/{len(self.dataset.get_train())} Loss: {loss.item():.6f}")

            average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.losses.append(average_epoch_loss)
        print(self.losses)
        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")

    def validate(self) -> None :
        self.model = self.model.eval()
        with torch.no_grad() :
            for i, (data, label) in enumerate(self.dataset.get_val()) :
                data = data.to(DEVICE)
                embedding_loss, x_hat, perplexity = self.model(data)
                recon_loss = torch.mean((x_hat - data)**2) / self.dataset.val_var()
                loss = recon_loss + embedding_loss

                if i % 10 == 0 :
                    print(f"Batch: {i+1}/{len(self.dataset.get_val())} Loss: {loss.item():.6f}")
                
    def plot(self, save = True) -> None :
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.title('Training Epochs against Loss for VQVAE')
        plt.xlabel('Epochs')
        plt.ylabel('Reconstruction Loss')
        plt.legend()
        plt.grid(True)

        if save :
            plt.savefig(self.savepath + '_training_loss.png')
        else :
            plt.show()

    def save(self, newpath = None) -> None :
        if newpath :
            torch.save(self.model.state_dict(), newpath)
        else :
            torch.save(self.model.state_dict(), self.savepath + '/vqvae.pth')