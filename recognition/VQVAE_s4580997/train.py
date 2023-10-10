import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from utils import Config
from dataset import Dataset
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from modules import GAN

class Trainer(ABC) :

    @abstractmethod
    def __init__(self, model: nn.Module, dataset: Dataset, config: Config) :
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
    
    def __init__(self, model: nn.Module, dataset: Dataset, config: Config) :

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
        
        self.losses = []
        self.accuracies = []

    def train(self) -> None :
        if self.dataset.train_unloaded() :
            self.dataset.load_train()
        
        self.model.train()
        start = time.time()
        for epoch in range(self.epochs) :
            epoch_loss = []
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

        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")

    @override
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

class TrainGAN(Trainer) :
    def __init__(self, model: nn.Module, dataset: Dataset, config: Config) :
        super().__init__(model, dataset, config)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)
        self.d_optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.g_optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.G_losses = []
        self.D_losses = []
        self.gan = GAN(features = 128)
        self.accuracies = []
    
    def train(self) -> None :
        pass

    
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
