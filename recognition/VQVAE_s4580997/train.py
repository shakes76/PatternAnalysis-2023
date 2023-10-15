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
from abc import ABC, abstractmethod

class Trainer(ABC) :
    def __init__(self, model: nn.Module, dataset: Dataset, lr=1e-3, wd=0, epochs=10, savepath='./models/vqvae') :
        self.savepath = savepath
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.model = model.to(DEVICE)
        self.dataset = dataset
    
    @abstractmethod
    def train(self) -> None :
        pass

    @abstractmethod
    def validate(self) -> None :
        pass

    @abstractmethod
    def plot(self, save = True, show = True) -> None :
        pass


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

class TrainGAN(Trainer) :
    def __init__(self, model: nn.Module, dataset: Dataset, lr=1e-3, wd=0, epochs=10, savepath='./models/gan') :
        super().__init__(model, dataset, lr, wd, epochs, savepath)

        self.d_optim = torch.optim.Adam(self.model.discriminator.parameters(), lr = self.lr)
        self.g_optim = torch.optim.Adam(self.model.generator.parameters(), lr = self.lr)
        self.criterion = nn.BCELoss()
        self.d_losses = list()
        self.g_losses = list()
        self.latent = 128

    def train(self) -> None :
        if self.dataset.train_unloaded() :
            self.dataset.load_train()
        
        self.model.train()
        start = time.time()
        for epoch in range(self.epochs) :
            g_epoch_loss = list()
            d_epoch_loss = list()
            for i, (data, label) in enumerate(self.dataset.get_train()) :
                data = data.to(DEVICE)
                self.d_optim.zero_grad()
                self.g_optim.zero_grad()

                noise = torch.randn(data.shape[0], self.latent, 1, 1).to(DEVICE)
                fake = self.model.generator(noise)
                
                real_pred = self.model.discriminator(data).reshape(-1)
                fake_pred = self.model.discriminator(fake).reshape(-1)

                real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
                fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                self.d_optim.step()

                output = self.model.discriminator(fake).reshape(-1)
                g_loss = self.criterion(output, torch.ones_like(output))
                g_loss.backward()

                self.g_optim.step()

                g_epoch_loss.append(g_loss.item())
                d_epoch_loss.append(d_loss.item())

                if (i % 10 == 0):
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {i+1}/{len(self.dataset.get_train())} Loss D: {D_loss.item():.6f}, Loss G: {G_loss.item():.6f}")
            
            self.g_losses.append(sum(g_epoch_loss) / len(g_epoch_loss))
            self.d_losses.append(sum(d_epoch_loss) / len(d_epoch_loss))  
        print(self.D_losses)
        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")

    def validate(self) -> None :
        pass
                
    def plot(self, save = True, show = True) -> None :
        plt.figure(figsize=(10, 5))
        plt.plot(self.d_losses, label='Discriminator Loss')
        plt.plot(self.g_losses, label='Generator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.title('GAN Training Loss')
        if save :
            plt.savefig(self.savepath + '_training_loss.png')
        if show :
            plt.show()
    
    def save(self, discriminator_path = None, generator_path = None) -> None :
        if discriminator_path and generator_path :
            torch.save(self.model.discriminator.state_dict(), discriminator_path)
            torch.save(self.model.generator.state_dict(), generator_path)
        else :
            torch.save(self.model.discriminator.state_dict(), self.savepath + '/generator.pth')
            torch.save(self.model.generator.state_dict(), self.savepath + '/discriminator.pth')