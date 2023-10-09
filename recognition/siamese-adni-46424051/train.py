##################################   train.py   ##################################
from dataset import DatasetTrain, DatasetTest
from modules import Model
from utils import Loss
import constants
from torchvision import transforms
import torch
from torch.nn import BCEWithLogitsLoss, PairwiseDistance
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

class Train():
    def __init__(self):
        self.net = Model()
        self.transforms = transforms.Compose([
            transforms.Resize((105,105)),
            transforms.ToTensor()
        ])
        self.loss_function = Loss()
        self.trainSet = DatasetTrain(constants.train_path, transforms=self.transforms)
        self.testSet = DatasetTest(constants.test_path, transforms=self.transforms, size=constants.test_size)
        self.trainLoader = DataLoader(self.trainSet, batch_size=constants.batch_size, shuffle=False)
        self.testLoader = DataLoader(self.testSet, batch_size=1, shuffle=False)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=constants.learning_rate)

    def train(self):
        if constants.cuda:
            self.net.cuda()

        losses = []

        for epoch in range(constants.epochs):
            for batch_num, (img1, img2, label) in enumerate(self.trainLoader):
                if constants.cuda:
                    img1, img2, label = img1.cuda(), img2.cuda(), label.cuda()
                self.optimiser.zero_grad()
                out1, out2 = self.net(img1, img2)
                loss = self.loss_function(out1, out2, label)
                loss.backward()
                self.optimiser.step()
                print("loss:", loss.item())
            losses.append(loss.item())
        plt.figure()
        plt.scatter(np.arange(0, 1, 1), losses)
        plt.savefig("loss.png")
        return self.net

    def test(self):
        correct, incorrect = 0, 0
        for _, (img1, img2, label) in enumerate(self.testLoader):
            if constants.cuda:
                img1, img2 = img1.cuda(), img2.cuda()
            out1, out2 = self.net(img1, img2)
            dis = torch.abs(out1 - out2)
            # difference = torch.nn.functional.pairwise_distance(out1, out2)
            # print(difference.item())
            print(dis)
            pred = np.argmax(dis.cpu().detach().numpy())
            if pred == 0:
                correct += 1
            else:
                incorrect += 1
            if label[0] == 0:
                label = "SAME"
            else:
                label = "DIFF"
            print(label)
        print(correct * 1.0 / (correct + incorrect))