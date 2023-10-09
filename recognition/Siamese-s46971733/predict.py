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

# Path that model is saved to and loaded from.
PATH = './resnet_net.pth'
CLAS_PATH = './clas_net.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

print("\n")

resnet = Resnet().to(device)
clas_net = classifer().to(device)

batch_size = 6

resnet.load_state_dict(torch.load(PATH))
clas_net.load_state_dict(torch.load(CLAS_PATH))

# Datasets and Dataloaders
testset = get_dataset(train=0, clas=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)
def visualise_images(images, labels, predictions, num_images=4):
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        plt.subplot(3, num_images, i * 3 + 1)
        plt.imshow(images[i][0].cpu())
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(3, num_images, i * 3 + 2)
        if labels[i] == 1:
            plt.text(0.5, 0.5, 'AD', horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, f'NC', horizontalalignment='center', verticalalignment='center')
        plt.title("True Label")
        plt.axis("off")

        plt.subplot(3, num_images, i * 3 + 3)
        if predictions[i] == 1:
            plt.text(0.5, 0.5, 'AD', horizontalalignment='center', verticalalignment='center')
        else:
            plt.text(0.5, 0.5, f'NC', horizontalalignment='center', verticalalignment='center')
        plt.title("Predicted Label")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

data = next(iter(testloader))

images = data[0].to(device)
labels = data[1].to(device)

outputs = resnet(images)
clas = clas_net(outputs)

m = nn.Threshold(0.5, 0)
outthresh = m(clas)

test = torch.softmax(clas, dim=1)

predicted_class = test.argmax(dim=1)

for i in range(batch_size):
    print(f"[Label: {labels[i]}] Output {i} is: {clas[i]}, and OutThresh {i} is {outthresh[i]}")
    print(f"Test {i} is {test[i]} and Predicted class is {predicted_class[i]}")

visualise_images(images, labels, predicted_class, num_images=batch_size)