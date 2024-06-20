import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.datasets as datasets
import torch.utils.data as util_data
from torch.utils.data import DataLoader, Dataset
import torch
import random
from PIL import Image
import dataset as ds
import module as md

#device configuration
device = torch.device("cuda")

#start training the model
def model_train(model, optimizer, epochs, loss_list, counter):
    model.train()
    criterion = md.ContrastiveLoss()
    for epoch in range(epochs):
        print("epoch", epoch+1)
        for count, (img1, img2, label) in enumerate(ds.trainloader):
            #passing the images and label to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            output1, output2 = model(img1, img2)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if count % 10 == 0:
                print(f"Epoch number {epoch+1}\n Current loss {loss.item()}\n")
                loss_list.append(loss.item())
                counter.append(count + (epoch * 40))

#Now test the model
def model_test(model):
    model.eval()
    print("begin testing")
    correct = 0
    total = 0
    for (img1, img2, label) in ds.testloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)
        output1, output2 = model(img1, img2)
        distance = torch.nn.functional.pairwise_distance(output1, output2)
        pred = torch.where(distance > 0.5, 0.0, 1.0)
        right = torch.where(label == pred, 1, 0)
        guesses = right.size(dim=0)
        total = total + guesses
        correct = correct + torch.sum(right).item()

    print(correct/total * 100)





        
    



