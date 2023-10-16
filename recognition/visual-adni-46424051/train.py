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
import time

class Train():
    def __init__(self):
        self.net = Model(shape=(1, 700, 700), patches=7, hidden_dim=8, blocks=2, heads=2, out_dim=2)
        self.transforms = transforms.Compose([
            transforms.Resize((700, 700)),
            transforms.ToTensor()
        ])
        self.criterion = CrossEntropyLoss()
        self.trainSet = DatasetTrain(constants.train_path, transforms=self.transforms)
        self.testSet = DatasetTest(constants.test_path, transforms=self.transforms)
        self.trainLoader = DataLoader(self.trainSet, batch_size=constants.batch_size, shuffle=True)
        self.testLoader = DataLoader(self.testSet, batch_size=constants.batch_size, shuffle=False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=constants.learning_rate)

    def train(self):
        start = time.time()
        if constants.cuda:
            self.net.cuda()
        
        lossy = []
        accuracies = []

        for epoch in range(constants.epochs):
            correct, total = 0, 0
            losses = 0.0
            for i, (x, y) in enumerate(self.trainLoader):
                # print("image: ", x)
                if constants.cuda:
                    x, y = x.cuda(), y.cuda()
                y1 = self.net(x)
                # print("y1: ", y1)
                # print("argmax y1: ", torch.argmax(y1).unsqueeze(0))
                # print("y: ", y)
                loss = self.criterion(y1.unsqueeze(0), y.long())
                losses += loss.detach().cpu().item() / len(self.trainLoader)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                # print(f'Image at index {i} is type {y} and guess is type {y1}')
                if torch.argmax(y1, dim=0) == y:
                    correct+=1
                total+=1
            print(f'Loss at epoch {epoch} is {losses} and accuracy is {correct * 1.0 / total}')
            lossy.append(losses)
            accuracies.append(correct * 1.0 / total)

        train_time = time.time() - start

        plt.figure()
        plt.suptitle(f'Loss plot - training done in {train_time}')
        plt.plot(lossy)
        plt.savefig("loss.png")

        plt.figure()
        plt.suptitle(f'Training accuracy plot - training done in {train_time}')
        plt.plot(accuracies)
        plt.savefig("accuracy.png")
    
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
