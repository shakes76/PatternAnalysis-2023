import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

class RawSiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()
        # Follow https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf -> Siamese Neural Networks for One-shot Image Recognition
        # first convolution layer
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=10),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2))

        self.model2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.model3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.model4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.model1(x)
        output = self.model2(output)
        output = self.model3(output)
        output = self.model4(output)

        return output

class SiameseModel(nn.Module): # may not need this

    def __init__(self):
        super(SiameseModel, self).__init__()
        # Follow https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf -> Siamese Neural Networks for One-shot Image Recognition
        # temporary separate to two class -> considered combine later
        self.base_model = RawSiameseModel()

        self.final_connect_layer = nn.Sequential(
            nn.Linear(4096, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img1, img2):
        output1 = self.base_model(img1)
        output2 = self.base_model(img2)
        
        output = torch.abs(output1 - output2)
        output = self.final_connect_layer(output)

        return output

class BinaryModelClassifier(nn.Module):

    def __init__(self, siamese_model):
        super(BinaryModelClassifier, self).__init__()

        self.siamese = siamese_model # siamese model

        # idea from the tutor -> use the siamese model to train the data to extract feature vector
        # train the classifier using the feature vector from the siamese model
        pass

    def forward(self, x):
        output = self.siamese(x) # feature vector from siamese model
        return output

class ContrastiveLossFunction(nn.Module):
    # custom loss function based on https://www.kaggle.com/code/robinreni/signature-classification-using-siamese-pytorch
    def __init__(self):
        super(ContrastiveLossFunction, self).__init__()
        self.margin = 0.2

    def forward(self, output1, output2, label): 
        output = torch.abs(output1 - output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(output, 2) + (label) * torch.pow(torch.clamp(self.margin - output, min=0.0), 2))

        return loss_contrastive
            