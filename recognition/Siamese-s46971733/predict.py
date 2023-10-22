# Imports
import random
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
from sklearn.manifold import TSNE
import seaborn as sns

from dataset import get_dataset
from modules import Resnet, Resnet34, classifier, Resnet3D

# Path that model is saved to and loaded from.
PATH = 'resnet_net_local30_6.pth'
CLAS_PATH = 'clas_net_local30_6.pth'
DATA_PATH = 'predict_data.png'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("No CUDA Found. Using CPU")

print("\n")

test = 1            # Testing On/Off
plot_loss = 0       # Plotting On/Off
visualisation = 1   # Visualisation of Brains On/Off
data_visual = 1     # Visualisation of Feature Vectors On/Off

resnet = Resnet3D().to(device)
clas_net = classifier().to(device)

batch_size = 8

resnet.load_state_dict(torch.load(PATH))
clas_net.load_state_dict(torch.load(CLAS_PATH))

# Datasets and Dataloaders
testset = get_dataset(train=0, clas=0)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
def visualise_images(images, labels, predictions, num_images=6, slice=0):
    plt.figure(figsize=(12, 6))

    for i in range(num_images):
        plt.subplot(3, num_images, i * 3 + 1)
        plt.imshow(images[i][0][slice].cpu())
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
    plt.savefig("./Images/PredictOutput.png")
    plt.show()

data = next(iter(testloader))

images = data[0].to(device)
labels = data[1].to(device)

outputs = resnet(images)
clas = clas_net(outputs)

if test == 1:
    print(">>> Testing Start")

    # Start timing.
    st = time.time()

    # Set Model to evaluation mode.
    clas_net.eval()

    # Gradient not required, improves performance.
    with torch.no_grad():
        correct = 0.0
        total = 0.0

        for i, data in enumerate(testloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            # Forward Pass
            features = resnet(inputs)
            output = clas_net(features)

            #output = torch.sigmoid(output)
            predicted = torch.round(output)

            # For each image in the batch -> as .size is [batch, channel, height, width]
            for index in range(inputs.size(0)):
                pred = predicted[index].cpu()
                true_val = labels[index].cpu()

                # Calculate dice coefficient and append to list.
                if pred == true_val:
                    correct += 1

                total += 1

        accuracy = correct/total

        print(f"Accuracy of the Model is {accuracy*100}%")

#######################################
# Resnet Data Visualisation with TSNE #
#######################################

trainloader_tsne = torch.utils.data.DataLoader(testset, batch_size=len(testloader.dataset),
                                               shuffle=True)

for i, data in enumerate(trainloader_tsne, 0):
    features_tsne = data[0].to(device)
    labels_tsne = data[1].to(device).cpu().numpy()

    if data_visual == 1:
        for param in resnet.parameters():
            param.requires_grad = False

        label_map = {0: "NC", 1: "AD"}
        features_tsne1 = resnet(features_tsne)

        train_tsne = TSNE().fit_transform(features_tsne1.cpu())

        print(train_tsne[: 0])

        plt.figure()
        sns.scatterplot(
            x=train_tsne[:, 0],
            y=train_tsne[:, 1],

            hue=[label_map[i] for i in labels_tsne])

        plt.title("training")
        plt.legend()
        plt.savefig(DATA_PATH)
        plt.show()


if visualisation == 1:
    # Randomly pick a batch of images.
    random_images = random.randrange(1, len(testloader))
    for i, data in enumerate(testloader, 0):
        if i == random_images:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = resnet(images)
            clas = clas_net(outputs)

            predicted_class = torch.round(clas)

            visualise_images(images, labels, predicted_class, slice=0)






