"""
Main driver script for training/evaluating model, saving/loading model,
testing the model, and saving the training/validation loss plots.

NOTES:
- Remember to add the filepaths for saving/loading model and loading the data sets
- Increase/decrease batch sizes depending on available GPU memory
    - Current batch sizes are configured for 8GB of GPU memory with a model of 4 base filters and the resize transform of (672, 1024)
    - Batch size for testing set can be larger than training/validation set to speed up testing phase
- When model filters are doubled from base level the batch size needs to be halved for training/validation set
"""

import torch
import torchvision.transforms as transforms
import time
from modules import ImprovedUNet
from dataset import CustomDataset
import numpy as np
import matplotlib.pyplot as plt

# File path for saving and loading model
filepath = "path to file\\ImprovedUNet.pt"

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Training Hyper-parameters
num_epochs = 20
learning_rate = 1e-3

#--------------
# DATA PREPROCESSING AND DATA LOADING
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
trainset = CustomDataset('path to file\\ISIC2018\\ISIC2018_Task1-2_Training_Input_x2',
                         "path to file\\ISIC2018\\ISIC2018_Task1_Training_GroundTruth_x2",
                         transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True)
total_step = len(train_loader) #used for counting overall steps for training loop in scheduler

validationset = CustomDataset("path to file\\ISIC2018\\ISIC2018_Task1-2_Validation_Input",
                         "path to file\\ISIC2018\\ISIC2018_Task1_Validation_GroundTruth",
                         transform=transform_validate)
validation_loader = torch.utils.data.DataLoader(validationset, batch_size=16, shuffle=True)
total_val_step = len(validation_loader)

testset = CustomDataset("path to file\\ISIC2018\\ISIC2018_Task1-2_Test_Input",
                         "path to file\\ISIC2018\\ISIC2018_Task1_Test_GroundTruth",
                         transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
total_test_step = len(test_loader)
# END OF DATA PREPROCESSING AND DATA LOADING
#--------------


#--------------
# SETUP FOR MODEL, TRAINING LOOP AND LOSS FUNCTION
#setup the model parameters and send to GPU
model = ImprovedUNet(in_channels=3, out_channels=1, base_n_filter=4)
model = model.to(device)

#model info
print("Model No. of Parameters:", sum([param.nelement() for param in model.parameters()]))
print(model)

# From https://github.com/pytorch/pytorch/issues/1249#issuecomment-305088398
def dice_loss(input, target):
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

#Adam Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
# END OF SETUP
#--------------


#--------------
# MODEL TRAINING PHASE
print("> Training")
trainingLossList = []
validationLossList = []
start = time.time() #time generation

# Training Loop
for epoch in range(num_epochs):
    model.train() #model set to evaluation mode
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

    # print the average training accuracy for the epoch
    print('Training Accuracy: {} %'.format(1 - (trainingLossAvg/total_step)))

    # append the average loss to the list to be used for plotting
    trainingLossList.append((trainingLossAvg/total_step))

    # Validation Loop
    model.eval() #set model to evaluation mode
    with torch.no_grad():
        validationLossAvg = 0
        for (images, masks) in validation_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            validationLossAvg += dice_loss(outputs, masks).detach().cpu().numpy()

        # print the validation accuracy as the dice coefficient which is (1 - dice loss)
        print('Validation Accuracy: {} %'.format(1 - (validationLossAvg/total_val_step)))
        validationLossList.append((validationLossAvg/total_val_step))

        # breaks out of the training loop early if validation dice coefficient equals or is greater than 0.8
        # this conditional is OPTIONAL and can be commented out if you want to let the training loop go to number of epochs set
        if 1 - (validationLossAvg/total_val_step) >= 0.8:
            break

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
# END OF MODEL TRAINING PHASE
#--------------


#--------------
# SAVE TRAINED MODEL AND THE TRAINING/VALIDATION LOSS PLOTS
torch.save(model, filepath) #save the model at the filepath

# plot the training loss across the epochs
plt.plot(trainingLossList, label="Training Loss")
plt.plot(validationLossList, label="Validation Loss")

plt.ylim(0,1) #ensures the range for the y-axis on the plot is within 0 and 1

plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.legend()

plt.savefig("training_loss.png") #saves plot in the same folder that train.py is located in
# END OF MODEL SAVING AND PLOT SAVING
#--------------


#--------------
# MODEL TESTING PHASE
print("> Testing")
loadedModel = torch.load(filepath) #load the model from the filepath
start = time.time() #time generation
loadedModel.eval() #set model to evaluation mode
with torch.no_grad():
    lossAvg = 0
    for (images, masks) in test_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        
        lossAvg += dice_loss(outputs, masks)

    # print the test accuracy as the dice coefficient which is (1 - dice loss)
    print('Test Accuracy: {} %'.format(1 - (lossAvg/total_test_step)))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total")
# END OF MODEL TESTING PHASE
#--------------
