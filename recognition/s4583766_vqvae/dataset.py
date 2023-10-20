'''
Loads the data and preprocesses it for use in the model.

Sophie Bates, s4583766.
'''

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image
from PIL import Image

class OasisDataset(Dataset):
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    def __init__(self, data_path):
        self.data_path = data_path
        self.images = os.listdir(data_path)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, x):
        img_path = self.data_path + self.images[x]
        image = Image.open(img_path)

        image = self.transform(image)
        return image

# Display a batch of training images
def show_batch(dl):
    imgs = next(iter(dl))
    img = make_grid(imgs[0])
    show_images(img)
    save_image(img, 'test.png')

# Save all images in one train_dl batch
def show_images(img, epoch):
    """
    Plot images grid of single batch
    """
    img = img.numpy()
    fig = plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    # save to 'gen_img' folder
    img_name = f'gen_img/s_epoch_{epoch}.png'
    save_image(img, img_name)
    print("Saving", img_name)

    plt.savefig(img_name)
    plt.clf()

# show_batch(train_dl)

def get_dataloaders(data_path_training, data_path_testing, data_path_validation, batch_size):
    """
    Loads the dataset and returns the training data loaders.
    """
    train_ds = OasisDataset(data_path_training)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    test_ds = OasisDataset(data_path_testing)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    val_ds = OasisDataset(data_path_validation)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=True)

    # show_batch(train_dl)
    # Calculate the variance of the data
    # data_variance = np.var(train_ds[0][0].numpy())
    # print(data_variance)

    return train_dl, test_dl, val_dl
