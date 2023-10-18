import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import time

from dataset import *
from modules import YOLO, compute_loss


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
epochs = 10
learning_rate=0.001
image_size = 416
batch_size = 10

#data - change directories as needed
mask_dir = '/content/drive/MyDrive/Uni/COMP3710/ISIC-2017_Training_Part1_GroundTruth/'
image_dir = '/content/drive/MyDrive/Uni/COMP3710/ISIC-2017_Training_Data/'
labels = '/content/drive/MyDrive/Uni/COMP3710/ISIC-2017_Training_Part3_GroundTruth.csv'
dataset = ISICDataset(image_dir, mask_dir, labels, image_size)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#Model
model = YOLO()
model.to(device)

#optimizer
optimizer = torch.optim.Adam(model.parameters(), learning_rate)

#learning rate schedule, using because SGD is dumb, adam has its own learning rate
total_step = len(dataloader)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,max_lr=learning_rate,steps_per_epoch=total_step, epochs=epochs)
#Train
model.train()
start = time.time()
for epoch in range(epochs):
  for i, (images, labels) in enumerate(dataloader):
    images = images.to(device)
    labels = labels.to(device)

    #Forward pass
    outputs = model(images)
    loss = compute_loss(outputs, labels, batch_size)

    #Backwards and optimize
    optimizer.zero_grad()
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

    if (i+1) % 50 == 0:
      print("Epoch [{}/{}], Step[{},{}] Loss: {:.5f}".format(epoch+1, epochs, i+1, total_step, loss.item()))

    scheduler.step()
end = time.time()
elapsed = end - start
print("Training took {} secs or {} mins.".format(elapsed, elapsed/60))
