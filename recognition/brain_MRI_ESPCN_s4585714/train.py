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

train_loader, valid_loader = dataset.ADNIDataLoader(root, mode='train')

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
    total_loss = 0
    for downscaleds, origs in train_loader: #load a batch
        downscaleds = downscaleds.to(device)
        origs = origs.to(device)

        # Forward pass
        outputs = model(downscaleds)
        loss = criterion(outputs, origs)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    total_loss = total_loss/len(train_loader)
    print ("Epoch [{}/{}], Loss: {:.5f}"
           .format(epoch+1, num_epochs, total_loss))
    train_loss.append(total_loss)
        
    # validation
    total_loss = 0
    with torch.no_grad():
        for downscaleds, origs in valid_loader: #load a batch
            downscaleds = downscaleds.to(device)
            origs = origs.to(device)
    
            # Forward pass
            outputs = model(downscaleds)
            loss = criterion(outputs, origs)
            
            total_loss += loss.item()

    total_loss = total_loss/len(train_loader)
    print ("Epoch [{}/{}], Loss: {:.5f}"
           .format(epoch+1, num_epochs, total_loss))
    valid_loss.append(total_loss)

    if smallest_valid_loss > total_loss:
        torch.save(model.state_dict(), 'model.pth')
        smallest_valid_loss = total_loss

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# plotting
plt.figure(1)
plt.plot(train_loss, label="Training Loss")
plt.plot(valid_loss, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.tight_layout()
plt.savefig("training_loss.png")