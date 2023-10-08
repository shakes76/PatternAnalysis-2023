##################################   train.py   ##################################
from dataset import DatasetTrain, DatasetTest
from modules import Model
import constants
from torchvision import transforms
import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

class Train():
    def __init__(self):
        self.net = Model()
        self.trainSet = DatasetTrain(constants.train_path)
        self.testSet = DatasetTest(constants.test_path, size=constants.train_iters)
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trainLoader = DataLoader(self.trainSet, batch_size=constants.batch_size, shuffle=False)
        self.testLoader = DataLoader(self.testSet, batch_size=constants.batch_size, shuffle=False)
        self.loss_function = BCEWithLogitsLoss(size_average=True)
        self.optimiser = torch.optim.Adam(self.net.parameters(), lr=constants.learning_rate)

    def train(self):
        if constants.cuda:
            self.net.cuda()
        self.net.train()

        losses = []
        loss = 0

        for batch_num, (img1, img2, label) in enumerate(self.trainLoader):
            print(batch_num)
            if batch_num == constants.train_iters:
                break
            if constants.cuda:
                img1, img2, label = Variable(img1.cuda()), Variable(img2.cuda()), Variable(label.cuda())
            else:
                img1, img2, label = Variable(img1), Variable(img2), Variable(label)
            
            self.optimiser.zero_grad()
            out = self.net.forward(img1, img2)
            loss += self.loss_function(out, label).item
            self.optimiser.step()
            losses.append(loss)

    def test(self):
        correct, incorrect = 0, 0
        for _, (img1, img2) in enumerate(self.testLoader):
            if constants.cuda:
                img1, img2 = img1.cuda(), img2.cuda()
            img1, img2 = Variable(img1), Variable(img2)
            out = self.net.forward(img1, img2).data.cpu().numpy()
            prediction = np.argmax(out)
            if prediction == 0:
                correct += 1
            else:
                incorrect += 1
        print (correct * 1.0 / (incorrect + correct))