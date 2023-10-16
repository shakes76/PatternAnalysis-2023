import torch
import torchvision.transforms as transforms
import time
from modules import ImprovedUNet
from dataset import CustomDataset
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# File path for saving and loading model
filepath = "path to file\\ImprovedUNet.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 2
learning_rate = 5e-3

#--------------
#Data
imageTransform_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.7083, 0.5821, 0.5360), (0.0969, 0.1119, 0.1261)),
                                           transforms.Resize((1024, 672))])
maskTransform_train = transforms.Compose([transforms.ToTensor(), transforms.Resize((1024, 672))])

imageTransform_test = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize((0.7083, 0.5821, 0.5360), (0.0969, 0.1119, 0.1261)),
                                           transforms.Resize((1024, 672))])
maskTransform_test = transforms.Compose([transforms.ToTensor(), transforms.Resize((1024, 672))])

trainset = CustomDataset('path to file\\ISIC2018\\ISIC2018_Task1-2_Training_Input_x2',
                         "path to file\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth_x2",
                         imageTransform=imageTransform_train,
                         maskTransform=maskTransform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
total_step = len(train_loader)

testset = CustomDataset("path to file\\ISIC2018\\ISIC2018_Task1-2_Test_Input",
                         "path to file\\ISIC2018\\ISIC2018_Task1_Test_GroundTruth",
                         imageTransform=imageTransform_test,
                         maskTransform=maskTransform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=True)

#Training
model = ImprovedUNet(in_channels=3, out_channels=1, base_n_filter=1)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=1e-5)

#Piecewise Linear Schedule
total_step = len(train_loader)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_step*num_epochs)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-1, total_steps=total_step*num_epochs)

#--------------
# Train the model
#"""
model.train()
print("> Training")
lossList = []
start = time.time() #time generation
for epoch in range(num_epochs):
    lossAvg = 0
    for i, (images, masks) in enumerate(train_loader): #load a batch
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        loss = dice_loss(outputs, masks)
        lossAvg += loss.detach().cpu().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
        scheduler.step()
    print(lossAvg/total_step)
    lossList.append((lossAvg/total_step))
end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
#"""

torch.save(model, filepath)

plt.plot(lossList, label="Training Loss")

plt.ylim(0,1)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("training_loss.png")

loadedModel = torch.load(filepath)

# Test the model
print("> Testing")
original = []
reconstruction = []
start = time.time() #time generation
loadedModel.eval()
with torch.no_grad():
    lossAvg = 0
    for (images, masks) in test_loader:
        batch_size = images.size(0)
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        lossAvg += dice_loss(outputs, masks)
    print('Test Accuracy: {} %'.format(1 - (lossAvg/total_step)))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
