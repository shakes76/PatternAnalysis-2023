# Contains the source code of the components of your model. Each component must be
# implementated as a class or a function
# Import all the necessary libraries
import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import random
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")
else:
    print("Using CUDA.")

def imshow(img,text=None,should_save=False):#for showing the data you loaded to dataloader
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_plot(iteration,loss):# for showing loss value changed with iter
    plt.plot(iteration,loss)
    plt.show()

class Config():
    training_dir = 'C:\\Users\\david\\OneDrive\\Documents\\0NIVERSITY\\2023\\SEM2\\COMP3710\\Project\\PatternAnalysis-2023\\recognition\\Siamese_45330212\\AD_NC\\train'
    testing_dir = 'C:\\Users\\david\\OneDrive\\Documents\\0NIVERSITY\\2023\\SEM2\\COMP3710\\Project\\PatternAnalysis-2023\\recognition\\Siamese_45330212\\AD_NC\\test'
    train_batch_size = 32
    train_number_epochs = 20

# --------------
# Model
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(28672*block.expansion, 2) # 28672 6144

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward_once(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11,stride=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5,stride=2,padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),
        )

        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(10752, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),

            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128,2)
        )

    def forward_once(self, x):
        # Forward pass
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        output1 = self.forward_once(input1)
        # forward pass of input 2
        output2 = self.forward_once(input2)
        return output1, output2

class CustomSiameseNetworkDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Create a list of image paths
        self.image_paths = []
        self.labels = []
        folders = os.listdir(root_dir)
        print("> Creating image paths")
        for i, folder1 in enumerate(folders):
            for j, folder2 in enumerate(folders):
                c = 0
                print("Folder:", folder1, folder2)
                if i == j:
                    label = 0  # Images from the same folder
                else:
                    label = 1  # Images from different folders

                folder1_path = os.path.join(root_dir, folder1)
                folder2_path = os.path.join(root_dir, folder2)

                folder1_images = os.listdir(folder1_path)
                folder2_images = os.listdir(folder2_path)

                for img1 in folder1_images:
                    c += 1
                    if c % 1000 == 0:
                        print("Count:", c)
                    img2 = random.choice(folder2_images)
                    while img1 == img2 and i == j:
                        print("FOUND SAME IMAGE - SHOULDN'T HAPPEN OFTEN")
                        img2 = random.choice(folder2_images)

                    img1_path = os.path.join(folder1_path, img1)
                    img2_path = os.path.join(folder2_path, img2)

                    self.image_paths.append((img1_path, img2_path))
                    self.labels.append(label)

        print("< Finished creating image paths. #Images:", len(self.image_paths))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img1_path, img2_path = self.image_paths[index]
        img1 = Image.open(img1_path).convert("RGB")
        img2 = Image.open(img2_path).convert("RGB")

        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return img1, img2, label

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on formula provided during symposium.
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        # if torch.isnan(loss_contrastive):
        #     print("Value is NaN")
        return loss_contrastive

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.11550809, 0.11550809, 0.11550809), (0.22545652, 0.22545652, 0.22545652)),
])

trainset = CustomSiameseNetworkDataset(root_dir=Config.training_dir, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=Config.train_batch_size, shuffle=True)

# testset = CustomSiameseNetworkDataset(root_dir=Config.testing_dir, transform=transform_train)
testset = datasets.ImageFolder(Config.testing_dir, transform=transform_train)
test_loader = torch.utils.data.DataLoader(testset, batch_size=Config.train_batch_size, shuffle=True)

# model = SiameseNetwork()
model = ResNet18()
model = model.to(device)

print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))

# Decalre Loss Function
criterion = ContrastiveLoss()
learning_rate = 1e-4
# optimizer = optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0.0005, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=Config.train_number_epochs)

# --------------
# Train the model
model.train()
print("> Training")
start = time.time() #time generation
counter = []
loss_history = []
iteration_number= 0
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
for epoch in range(0,Config.train_number_epochs):
    print(enumerate(train_loader,0).__sizeof__(), Config.train_batch_size)
    for i, data in enumerate(train_loader,0):
        #Produce two sets of images with the label as 0 if they're from the same file or 1 if they're different
        img1, img2, labels = data
        img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

        # Forward pass
        output1, output2 = model(img1, img2)
        # print("Out1len", output1.size(), "Out2len", output1.size(), "labelssize", labels.size())
        loss_contrastive = criterion(output1, output2, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss_contrastive.backward()
        optimizer.step()
        if i % 10 == 0:
            print("i:", i, "/", int(43040 / Config.train_batch_size))
        if i % 50 == 0:
            print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.item()))
            iteration_number += 50
            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
        scheduler.step()

    #test the network after finish each epoch, to have a brief training result.
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device) , label.to(device)
            output, _ = model(img, img)
            _, predicted = torch.max(output.data, 1)
            total_val += label.size(0)
            correct_val += (predicted == label).sum().item()

    print('Accuracy of the network on the', total_val, ': %d %%' % (100 * correct_val / total_val))
    show_plot(counter, loss_history)


torch.save(model.state_dict(), "model.pt")
print("Model Saved Successfully")

# --------------
# Test the model
print("> Testing")
start = time.time() #time generation
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for img, label in test_loader:
        img, label = img.to(device) , label.to(device)
        output, _ = model(img, img)
        _, predicted = torch.max(output.data, 1)
        total_val += label.size(0)
        correct_val += (predicted == label).sum().item()

    print('Test Accuracy: {} %'.format(100 * correct / total))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")

print('END')