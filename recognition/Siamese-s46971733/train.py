# Imports
import time
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import cv2
import os
from dataset import get_dataset
from modules import Resnet, Resnet34, classifer

# Toggles.
train = 1
test = 1
plot_loss = 1

# Path that model is saved to and loaded from.
PATH = './resnet_net.pth'
CLAS_PATH = './clas_net.pth'

# Path that training loss is saved to.
PLOT_PATH = './training_loss.png'

# Hyperparameters
num_epochs = 5
batch_size = 32
learning_rate = 0.001
res_learning_rate = 0.0001

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

print("\n")

# Datasets and Dataloaders
trainset = get_dataset(train=1, clas=0)
testset = get_dataset(train=0, clas=0)
trainset_clas = get_dataset(train=1, clas=1)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
trainloader_clas = torch.utils.data.DataLoader(trainset_clas, batch_size=batch_size, shuffle=True)

# Model.
resnet = Resnet().to(device)
clas_net = classifer().to(device)

# Optimizer

criterion_net = nn.CrossEntropyLoss()
criterion = nn.TripletMarginLoss()
criterion_class = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet.parameters(), lr=res_learning_rate, weight_decay=0.0001)
class_optimizer = optim.Adam(resnet.parameters(), lr=learning_rate)

# Future spot for Scheduler?



#

loss_list = []
class_loss_list = []

if train == 1:
    # Training Model
    resnet.train()
    print(f">>> Training \n")
    # Start timing.
    st = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0

        # Loop over every batch in data loader.
        for i, data in enumerate(trainloader, 0):
            # Extract data and transfer to GPU.
            anchor = data[0].to(device)
            positive = data[1].to(device)
            negative = data[2].to(device)

            # Zero the gradients -- Ensuring gradients not accumulated
            #                       across multiple training iterations.
            optimizer.zero_grad()

            # Forward Pass
            anchor_out = resnet(anchor)
            positive_out = resnet(positive)
            negative_out = resnet(negative)

            # Calculate Loss with Triplet Loss.
            loss = criterion(anchor_out, positive_out, negative_out)

            # Compute gradient with respect to model.
            loss.backward()

            # Optimizer step - Update model parameters.
            optimizer.step()

            # Keep track of running loss.
            running_loss += loss.item()

            # Print Loss Info while training.
            if (i + 1) % 10 == 0:
                print(f'[Epoch {epoch + 1}/{num_epochs}, {i + 1:5d}] - Loss: {running_loss / 10:.5f}')
                running_loss = 0.0

            loss_list.append(loss.item())

        ###############

    # Train Classifier.

    for param in resnet.parameters():
        param.requires_grad = False

    print(f"\n>>> Training Classifier \n")
    # Start timing.
    class_st = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0

        clas_net.train()
        resnet.eval()

        # Loop over every batch in data loader.
        for i, data in enumerate(trainloader_clas, 0):
            # Extract data and transfer to GPU.
            inputs = data[0].to(device)
            labels = data[1].to(device)

            # Zero the gradients -- Ensuring gradients not accumulated
            #                       across multiple training iterations.
            class_optimizer.zero_grad()

            # Forward Pass
            res_output = resnet(inputs)
            output = clas_net(res_output)

            # Calculate Loss with Cross Entropy.
            loss = criterion_class(output, labels)
            # Compute gradient with respect to model.
            loss.backward()

            # Optimizer step - Update model parameters.
            class_optimizer.step()

            # Keep track of running loss.
            running_loss += loss.item()

            # Print Loss Info while training.
            if (i + 1) % 10 == 0:
                print(f'[Training Classifier][Epoch {epoch + 1}/{num_epochs}, {i + 1:5d}] - Loss: {running_loss / 10:.5f}')
                running_loss = 0.0

            class_loss_list.append(loss.item())

    print(">>> Training Finished.")
    elapsed = time.time() - st
    print(f"\nTraining took: {elapsed}s to complete, or {elapsed/60} minutes.\n")

    # Save trained model for later use.
    torch.save(resnet.state_dict(), PATH)
    torch.save(clas_net.state_dict(), CLAS_PATH)

else:
    print("Training was disabled. \nLoading model from path.")
    resnet.load_state_dict(torch.load(PATH))
    clas_net.load_state_dict(torch.load(CLAS_PATH))

if test == 1:
    print(">>> Testing Start")

    # Start timing.
    st = time.time()

    # Set Model to evaluation mode.
    resnet.eval()

    for i, data in enumerate(testloader, 0):
        inputs, labels = data[0].to(device), data[1]

        #print(f"Inputs are: {inputs}")

        predicted = resnet(inputs)

        #print(f"Outputs are: {F.softmax(predicted)}")

# Plot the loss over the many iterations of training.
if plot_loss == 1:
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.title("Resnet Loss")
    plt.plot(loss_list)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.subplot(2, 1, 2)
    plt.title("Classifier Loss")
    plt.plot(class_loss_list)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.savefig(PLOT_PATH)
    plt.show()
