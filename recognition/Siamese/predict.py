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
import module as md
import train as tr
import dataset as ds

if __name__ == '__main__':

    loader = DataLoader(ds.siamese_train,
                        shuffle=True,
                        batch_size=8)
    
    image1, image2, label = next(iter(loader))
    concatenated = torch.cat((image1, image2),0)

    img = torchvision.utils.make_grid(concatenated).numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    print(label.numpy().reshape(-1))

    #device configuration
    device = torch.device("cuda")

    #Define parameters for model
    layer = "VGG13"
    in_channels = 1
    classes = 2
    epochs = 5
    learning_rate = 1e-5  

    model = md.Siamese(layers=layer, in_channels=in_channels, classes=classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    loss = []
    counter = []

    tr.model_train(model, optimizer, epochs, loss, counter)
    print(loss)
    print(counter)
    plt.plot(counter, loss)
    plt.show()
    tr.model_test(model)

