import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from modules import Siamese
from utils import Config
from dataset import Dataset

class Train() :
    
    def __init__(self, model: nn.Module, dataset: Dataset, config: Config) :

        # Optimisation parameters
        self.lr = config.lr
        self.wd = config.wd
        self.epochs = config.epochs
        self.device = config.device

        self.model = model.to(self.device)
        self.dataset = dataset
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.criterion = nn.BCEWithLogitsLoss().to(self.device)

    def train(self) -> None :
        self.dataset.load_train()
        self.model.train()

        for epoch in range(self.epochs) :
            start = time.time()
            for batch_idx, (data, target) in enumerate(self.dataset.get_train()) :
                self.optimiser.zero_grad()
                output = self.model(data[0], data[1])
                loss = self.criterion(output.squeeze(), target)
                loss.backward()
                self.optimiser.step()
                if batch_idx % 10 == 0 :
                    print(f"Epoch: {epoch+1}/{self.epochs} Batch: {batch_idx+1}/{len(self.dataset.get_train())} Loss: {loss.item():.6f}")
            end = time.time()
            print(f"Time: {end-start:.2f}s")

    def test(self) -> None :
        self.dataset.load_test()
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad() :
            for batch_idx, (data, target) in enumerate(self.dataset.get_test()) :
                output = self.model(data[0], data[1])
                test_loss += self.criterion(output.squeeze(), target).item()
                pred = torch.round(torch.sigmoid(output))
                correct += (pred == target).float().sum()
        test_loss /= len(self.dataset.dataset)
        print(f"Test loss: {test_loss:.4f}, Accuracy: {correct}/{len(self.dataset.dataset.get_test())} ({100.*correct/len(self.dataset.get_test()):.0f}%)")
