import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from dataset import load_test
import time
from modules import ImprovedUNET
from dataset import load_test
from utilities import DiceLoss, Wandb_logger
import os
import matplotlib.pyplot as plt

#Using the model. Visual indication of results and performance

model = ImprovedUNET(3,16)

root = 'recognition/Tord_Improved_UNet'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state = torch.load(os.path.join(root,'model.pth'), map_location=device)

model.load_state_dict(state)

criterion = DiceLoss()

logger = Wandb_logger(model, criterion, config={"lr_init": 5e-4, "weight_decay": 1e-5})

dataLoader = DataLoader(load_test(), batch_size=2, shuffle=True, num_workers=1)

with torch.no_grad():
     for i, data in enumerate(dataLoader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # forward + backward + optimize
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            accuracy = criterion.accuracy(outputs, labels)
            logger.log_test(accuracy, loss)
            plt.close('all')
            image_tensor = outputs.cpu().detach()
            ground = labels.cpu().detach()
            image_array = image_tensor.numpy()
            ground_array = ground.numpy()

            # Display the stacked images side by side
            plt.figure(figsize=(15, 7))  # You can adjust the figure size as needed

            plt.subplot(1, 2, 1)
            plt.imshow(image_array[0, 0], cmap='gray')  # Display the first image in image_tensor
            plt.title("Prediction")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(ground_array[0, 0], cmap='gray')  # Display the first image in ground
            plt.title("Ground Truth")
            plt.axis('off')
            plt.show()

