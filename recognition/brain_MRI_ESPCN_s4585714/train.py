"""
Author: Zach Harbutt S4585714
contains source code for training, validating, testing and saving model. Plots various metrics

ref: https://keras.io/examples/vision/super_resolution_sub_pixel/#build-a-model
"""

import torch
import torch.nn as nn
import modules
import dataset
import matplotlib.pyplot as plt
import time
from skimage.metrics import peak_signal_noise_ratio
from torchvision import transforms
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU")

# Hyper-parameters
num_epochs = 100
learning_rate = 0.001
root = '/home/groups/comp3710/ADNI/AD_NC'

train_loader, valid_loader = dataset.ADNIDataLoader(root, mode='train')
test_loader = dataset.ADNIDataLoader(root, mode='test')

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
    print ("Epoch [{}/{}], Training Loss: {:.5f}"
           .format(epoch+1, num_epochs, total_loss))
    train_loss.append(total_loss)
        
    # validation
    total_loss = 0
    with torch.no_grad():
        model.eval()
        for downscaleds, origs in valid_loader: #load a batch
            downscaleds = downscaleds.to(device)
            origs = origs.to(device)
    
            # Forward pass
            outputs = model(downscaleds)
            loss = criterion(outputs, origs)
            
            total_loss += loss.item()

    total_loss = total_loss/len(valid_loader)
    print ("Epoch [{}/{}], Validation Loss: {:.5f}"
           .format(epoch+1, num_epochs, total_loss))
    valid_loss.append(total_loss)

    if smallest_valid_loss > total_loss:
        torch.save(model.state_dict(), 'model.pth')
        smallest_valid_loss = total_loss

end = time.time()
elapsed = end - start
print("Training took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# plot training
plt.figure(num=1)
plt.plot(train_loss, label="Training Loss")
plt.plot(valid_loss, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss over Epochs")
plt.tight_layout()
plt.savefig("training_loss.png")

# load the model state
model.load_state_dict(torch.load('model.pth'))

# Test the model
print("> Testing")
start = time.time() #time generation
toPil = transforms.ToPILImage()
psnr_preds = []
psnr_downscaleds = []
model.eval()
with torch.no_grad():
    total_psnr_downscaled = 0
    total_psnr_pred = 0
    for i, (downscaled, orig) in enumerate(test_loader):
        downscaled = downscaled.to(device)
        orig = orig.to(device)

        output = model(downscaled)
        
        downscaled = toPil(downscaled[0])
        downscaled = downscaled.resize((256, 240))
        downscaled = np.array(downscaled)
        orig = toPil(orig[0])
        orig = np.array(orig)
        output = toPil(output.squeeze())
        output = np.asarray(output)
        
        psnr_downscaled = peak_signal_noise_ratio(orig, downscaled)
        psnr_pred = peak_signal_noise_ratio(orig, output)
        
        total_psnr_downscaled += psnr_downscaled
        total_psnr_pred += psnr_pred
        psnr_downscaleds.append(psnr_downscaled)
        psnr_preds.append(psnr_pred)

        
    print('Avg. PSNR of lowres images is: {}'.format(total_psnr_downscaled/ len(psnr_downscaleds)))
    print('Avg. PSNR of reconstructions is: {}'.format(total_psnr_pred/ len(psnr_preds)))
end = time.time()
elapsed = end - start
print("Testing took " + str(elapsed) + " secs or " + str(elapsed/60) + " mins in total") 

# plot testing
plt.figure(num=2)
plt.plot(psnr_preds, label="Reconstruction PSNR")
plt.plot(psnr_downscaleds, label="Lowres Image PSNR")
plt.legend()
plt.xlabel("Sample")
plt.ylabel("PSNR")
plt.title("Lowres Image and Reconstruction PSNR")
plt.tight_layout()
plt.savefig("testing_psnr.png")

print('END')
