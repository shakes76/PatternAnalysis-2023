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
from abc import ABC, abstractmethod
from modules import GAN

class Trainer(ABC) :

    @abstractmethod
    def __init__(self, model: nn.Module, dataset: Dataset, config) :
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        self.device = config.device
        self.model = model.to(self.device)
        self.dataset = dataset
        
        self.losses = []

    @abstractmethod
    def train(self) -> None :
        pass

    def plot(self, save = True) -> None :
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.title('Training Epochs against Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        if save :
            plt.savefig(self.savepath + '_training_loss.png')
        else :
            plt.show()
    
    def save(self,) -> None :
        pass
        

class TrainVQVAE() :
    
    def __init__(self, model: nn.Module, dataset: Dataset, config) :

        self.savepath = config.savepath
        # Optimisation parameters
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        self.device = config.device
        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        
        self.losses = list()

    def train(self) -> None :
        if self.dataset.train_unloaded() :
            self.dataset.load_train()
        
        self.model.train()
        start = time.time()
        for epoch in range(self.epochs) :
            epoch_loss = list()
            for i, (data, label) in enumerate(self.dataset.get_train()) :
                self.optimiser.zero_grad()

                embedding_loss, x_hat, perplexity = self.model(data)
                recon_loss = torch.mean((x_hat - data)**2) / self.dataset.train_var()
                loss = recon_loss + embedding_loss

                loss.backward()
                self.optimiser.step()

                epoch_loss.append(loss.item())

                if i % 10 == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {i+1}/{len(self.dataset.get_train())} Loss: {loss.item():.6f}")

            average_epoch_loss = sum(epoch_loss) / len(epoch_loss)
            self.losses.append(average_epoch_loss)
        print(self.losses)
        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")

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

class TrainGAN() :
    def __init__(self, model: nn.Module, dataset: Dataset, config) :
        self.savepath = config.savepath
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs

        self.device = config.device
        self.model = model.to(self.device)
        self.dataset = dataset

        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.gan = GAN(features = 128)
        self.d_optimiser = torch.optim.Adam(self.gan.discriminator.parameters(), lr=self.lr, weight_decay=self.wd)
        self.g_optimiser = torch.optim.Adam(self.gan.generator.parameters(), lr=self.lr, weight_decay=self.wd)
        self.G_losses = list()
        self.D_losses = list()
        self.latent = 128
    
    
    def train(self):
        self.gan.generator.train()
        self.gan.discriminator.train()
        start = time.time()
        for epoch in range(self.epochs):
            G_epoch_loss = list()
            D_epoch_loss = list()
            for i, (data, _) in enumerate(self.dataset.get_train()):
                real_data = data.to(self.device)
                batch_size = real_data.size(0)

                #Generate fake image to pass through to model
                noise = torch.randn(batch_size, self.latent, 1, 1).to(self.device)
                fake_img = self.gan.generator(noise)

                D_real = self.gan.discriminator(real_data).reshape(-1)
                D_real_loss = self.criterion(D_real, torch.ones_like(D_real))
                D_fake = self.gan.discriminator(fake_img.detach()).reshape(-1)
                D_fake_loss = self.criterion(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_fake_loss + D_real_loss) / 2
                self.gan.discriminator.zero_grad()
                D_loss.backward()
                self.d_optimiser.step()

                out = self.gan.discriminator(fake_img).reshape(-1)
                G_loss = self.criterion(out, torch.ones_like(out))
                self.gan.generator.zero_grad()
                G_loss.backward()
                self.g_optimiser.step()

                G_epoch_loss.append(G_loss.item())
                D_epoch_loss.append(G_loss.item())
                if (i % 10 == 0):
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {i+1}/{len(self.dataset.get_train())} Loss D: {D_loss.item():.6f}, Loss G: {G_loss.item():.6f}")
            
            self.G_losses.append(sum(G_epoch_loss) / len(G_epoch_loss))
            self.D_losses.append(sum(D_epoch_loss) / len(D_epoch_loss))  

        print(self.D_losses)
        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")
    
    def plot(self, save = True) -> None :
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss")
        plt.plot(self.G_losses, label="Generator")
        plt.plot(self.D_losses, label="Discriminator")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        if save :
            plt.savefig(self.savepath + '_gan_loss.png')
        else :
            plt.show()
    
    def save(self, discriminator_path = None, generator_path = None) -> None :
        if discriminator_path and generator_path :
            torch.save(self.gan.discriminator.state_dict(), discriminator_path)
            torch.save(self.gan.generator.state_dict(), generator_path)
        else :
            torch.save(self.gan.discriminator.state_dict(), self.savepath + '/generator.pth')
            torch.save(self.gan.generator.state_dict(), self.savepath + '/discriminator.pth')
