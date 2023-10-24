# showing example usage of your trained model. Print out any results and / or provide visualisations where applicable

from dataset import ISICDataset, transform
from modules import UNet
import torch
import torch.optim as optim


# Hyper-parameters
num_epochs = 100

def dice_loss(predicted, target, epsilon=1e-5):
    pass

model = UNet(6, 3)
optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.985)

for epoch in range(num_epochs):
    model.train() # Switch to training mode
    for images, masks in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        
    scheduler.step() # Adjust learning rate
