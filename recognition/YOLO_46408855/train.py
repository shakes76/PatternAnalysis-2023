import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import time

from dataset import *
from modules import YOLO


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
epochs = 10
learning_rate=0.001

#data - obviously will change when dataset.py is ready
path = '/content/drive/MyDrive/Uni/'
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)), #mean, SD
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, padding_mode='reflect')
    ])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)) #mean, SD
    ])

trainset = torchvision.datasets.CIFAR10(
    root=path+'data/cifar10', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root=path+'data/cifar10', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

#Model
model = YOLO()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

#learning rate schedule, using because SGD is dumb, adam has its own learning rate
total_step = len(train_loader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,steps_per_epoch=total_step, epochs=epochs)

#Train
model.train()
start = time.time()
for epoch in range(epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = images.to(device)
    labels = labels.to(device)

    #Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    #Backwards and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i+1) % 100 == 0:
      print("Epoch [{}/{}], Step[{},{}] Loss: {:.5f}".format(epoch+1, epochs, i+1, total_step, loss.item()))

    scheduler.step()
end = time.time()
elapsed = end - start
print("Training took {} secs or {} mins.".format(elapsed, elapsed/60))
