"""
Author: Zach Harbutt S4585714
contains source code for training, validating, testing and saving model. Plots various metrics

ref: https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
"""

import torch
import torch.nn as nn
import modules
import dataset
import matplotlib as plt
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001
root = 'AD_NC'

train_loader = dataset.ADNIDataLoader(root, mode='train')
valid_loader = dataset.ADNIDataLoader(root, mode='valid')

model = modules.ESPCN()
model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#--------------
# Train the model
train_loss = []
valid_loss = []
smallest_valid_loss = float('inf')
model.train()
print("> Training")
start = time.time() #time generation
for epoch in range(num_epochs):
    # training
    for i, (downscaleds, origs) in enumerate(train_loader): #load a batch
        downscaleds = downscaleds.to(device)
        origs = origs.to(device)

        # Forward pass
        outputs = model(downscaleds)
        loss = criterion(outputs, origs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ("Epoch [{}/{}], Loss: {:.5f}"
           .format(epoch+1, num_epochs, loss.item()))
    train_loss.append(loss.item())
        
    # validation
    for i, (downscaleds, origs) in enumerate(valid_loader): #load a batch
        downscaleds = downscaleds.to(device)
        origs = origs.to(device)

        # Forward pass
        outputs = model(downscaleds)
        loss = criterion(outputs, origs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ("Epoch [{}/{}], Loss: {:.5f}"
           .format(epoch+1, num_epochs, loss.item()))
    valid_loss.append(loss.item())

    if smallest_valid_loss > loss.item():
        torch.save(model.state_dict(), 'model.pth')
        smallest_valid_loss = loss.item()

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# plotting
plt