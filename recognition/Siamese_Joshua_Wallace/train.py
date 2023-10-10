import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from modules import Siamese
from utils import Config
from dataset import Dataset
import matplotlib.pyplot as plt

class Train() :
    
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
            epoch_loss = 0
            for i, (data, target) in enumerate(self.dataset.get_train()) :
                self.optimiser.zero_grad()
                output = self.model(data[0], data[1])
                loss = self.criterion(output, target.view(-1, 1))
                loss.backward()
                self.optimiser.step()
                epoch_loss += loss.item()
                if i % 10 == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {i+1}/{len(self.dataset.get_train())} Loss: {loss.item():.6f}")
            self.losses.append(epoch_loss / len(self.dataset.get_train()))
        end = time.time()
        print(f"Total Time for training: {end - start:.2f}s")
    
    def test(self) -> None :
        if self.dataset.test_unloaded() :
            self.dataset.load_test()
        
        self.model.eval()
        test_loss = 0
        correct = 0
        total_samples = 0
        start = time.time()
        with torch.no_grad() :
            for i, (data, target) in enumerate(self.dataset.get_test()) :
                output = self.model(data[0], data[1])
                test_loss += self.criterion(output.squeeze(), target).item()
                pred = torch.round(torch.sigmoid(output)).squeeze()
                correct += (pred == target.squeeze()).float().sum().item()
                total_samples += target.size(0)
        test_loss /= len(self.dataset.get_test())
        end = time.time()
        print(f"Total Time for Testing: {end - start:.2f}s")
        print(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{total_samples} ({100.*correct/total_samples:.0f}%)")

    def train_with_interval_testing(self) -> None :
        if self.dataset.train_unloaded() :
            self.dataset.load_train()

    def plot_loss(self) -> None :
        plt.figure(figsize=(10, 5))
        plt.plot(self.losses, label='Loss')
        plt.title('Training Epochs against BCE Loss')
        plt.xlabel('Epochs')
        plt.ylabel('BCE Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.savepath + '_training_loss.png')
    
    def plot_accuracies(self) -> None :
        pass