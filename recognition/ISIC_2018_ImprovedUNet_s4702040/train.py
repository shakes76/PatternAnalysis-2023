"""

"""

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
filepath = "ImprovedUNet.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 2
learning_rate = 1e-4

#--------------
#Data
transform_train = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((672, 1024),
                                                             antialias=True)])

transform_validate = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((672, 1024),
                                                             antialias=True)])

transform_test = transforms.Compose([transforms.ToTensor(),
                                           transforms.Resize((672, 1024),
                                                             antialias=True)])

# Load the datasets from the filepaths and put them into the dataloaders
trainset = CustomDataset('filepath\\ISIC2018\\ISIC2018_Task1-2_Training_Input_x2',
                         "filepath\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth_x2",
                         transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
total_step = len(train_loader)

validationset = CustomDataset("filepath\\ISIC2018\\ISIC2018_Task1-2_Validation_Input",
                         "filepath\\ISIC2018\\ISIC2018_Task1_Validation_GroundTruth",
                         transform=transform_validate)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=8, shuffle=True)

testset = CustomDataset("filepath\\ISIC2018\\ISIC2018_Task1-2_Test_Input",
                         "filepath\\ISIC2018\\ISIC2018_Task1_Test_GroundTruth",
                         transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

#Training
model = ImprovedUNet(in_channels=3, out_channels=1, base_n_filter=4)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

# From: 
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

#Piecewise Linear Schedule
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-1, total_steps=total_step*num_epochs)

#--------------
# Train the model
print("> Training")
trainingLossList = []
validationLossList = []
start = time.time() #time generation
for epoch in range(num_epochs):
    model.train()
    trainingLossAvg = 0
    for i, (images, masks) in enumerate(train_loader): #load a batch
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass
        outputs = model(images)

        loss = dice_loss(outputs, masks)
        trainingLossAvg += loss.detach().cpu().numpy()

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.5f}"
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
        scheduler.step()


    # print the average loss for the epoch
    print(trainingLossAvg/total_step)

    # append the average loss to the list to be used for plotting
    trainingLossList.append((trainingLossAvg/total_step))
    #" ""
    model.eval()
    with torch.no_grad():
        validationLossAvg = 0
        for (images, masks) in validation_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            validationLossAvg += dice_loss(outputs, masks).detach().cpu().numpy()

        # print the validation accuracy as the dice coefficient which is (1 - dice loss)
        print('Validation Accuracy: {} %'.format(1 - (validationLossAvg/total_step)))
        validationLossList.append((validationLossAvg/total_step))
        if 1 - (validationLossAvg/total_step) >= 0.8:
            break
    #" ""

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
#" ""

torch.save(model, filepath) #save the model at the filepath
#" ""
# plot the training loss across the epochs
plt.plot(trainingLossList, label="Training Loss")
plt.plot(validationLossList, label="Validation Loss")

plt.ylim(0,1)

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("training_loss.png")

#" ""

# Test the model
#"""
print("> Testing")

loadedModel = torch.load(filepath) #load the model from the described filepath
start = time.time() #time generation
loadedModel.eval()
with torch.no_grad():
    lossAvg = 0
    for (images, masks) in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        lossAvg += dice_loss(outputs, masks)

    # print the test accuracy as the dice coefficient which is (1 - dice loss)
    print('Test Accuracy: {} %'.format(1 - (lossAvg/total_step)))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
#"""
