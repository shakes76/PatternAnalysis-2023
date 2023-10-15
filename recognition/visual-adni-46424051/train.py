##################################   train.py   ##################################

from dataset import DatasetTrain, DatasetTest
from modules import Model
import constants
from torchvision import transforms
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Train():
    def __init__(self):
        self.net = Model(shape=(1, 28, 28), patches=7, hidden_dim=8, blocks=2, heads=2, out_dim=2)
        self.transforms = transforms.Compose([
            transforms.Resize((28,28)),
            transforms.ToTensor()
        ])
        self.criterion = CrossEntropyLoss()
        self.trainSet = DatasetTrain(constants.train_path, transforms=self.transforms)
        self.testSet = DatasetTest(constants.test_path, transforms=self.transforms, size=constants.test_size)
        self.trainLoader = DataLoader(self.trainSet, batch_size=constants.batch_size, shuffle=True)
        self.testLoader = DataLoader(self.testSet, batch_size=constants.batch_size, shuffle=False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=constants.learning_rate)

    def train(self):
        if constants.cuda:
            self.net.cuda()

        for epoch in range(constants.epochs):
            losses = 0.0
            for i, (x, y) in enumerate(self.trainLoader):
                print("image: ", x)
                if constants.cuda:
                    x, y = x.cuda(), y.cuda()
                y1 = self.net(x)
                # loss = self.criterion(y1, y)
                # losses += loss.detach().cpu().item() / len(self.trainLoader)
                self.optimiser.zero_grad()
                # loss.backward()
                self.optimiser.step()
                print(f'Image at index {i} is type {y} and guess is type {y1}')
            print(f'Loss at epoch {epoch} is {losses}')
    
    def test(self):
        with torch.no_grad():
            correct, total = 0.0, 0.0
            for i, (x, y) in enumerate(self.testLoader):
                if constants.cuda:
                    x, y = x.cuda(), y.cuda()
                y1 = self.net(x)
                correct += torch.sum(torch.argmax(y1, dim=0) == y).detach().cpu().item()
                total += len(x)
                print(f'Image at index {i} is type {y} and guess is type {y1}')
            print(f'Accuracy is {correct / total * 100}%')