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
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not Found. Using CPU...")

# Setup file paths
PATH = os.getcwd() + '/'
DATA_PATH_TRAINING_RANGPUR = '/home/groups/comp3710/OASIS'
DATA_PATH_TRAINING_LOCAL = PATH + 'test_img/'
BATCH_SIZE = 32
# Set the mode to either 'rangpur' or 'local' for testing purposes
mode = 'rangpur'

if mode == 'rangpur':
    DATA_PATH_TRAINING = DATA_PATH_TRAINING_RANGPUR
elif mode == 'local':
    DATA_PATH_TRAINING = DATA_PATH_TRAINING_LOCAL


# Perform tranformations on the data
transform_train = transforms.Compose([
    transforms.ToTensor()
])

train_ds = ImageFolder(DATA_PATH_TRAINING, transform=transform_train)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# Display a batch of training images
def show_batch(dl):
    imgs = next(iter(dl))
    img = make_grid(imgs[0])
    show_images(img)
    save_image(img, 'test.png')

# Save all images in one train_dl batch
def show_images(img):
    """
    Plot images grid of single batch
    """
    img = img.numpy()
    fig = plt.imshow(np.transpose(img, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

show_batch(train_dl)