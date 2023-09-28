import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from dataset import ADNIDataset  # Import your custom dataset class
from modules import Generator, Discriminator  # Import your generator and discriminator models

# Set up your training parameters, model, and optimizer
# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1 # black and white images

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Initialize your custom dataset and data loader
data_root = './AD_NC'  # Adjust this to your dataset directory
batch_size = 128
# Modify the transformation to convert grayscale images to tensors
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.CenterCrop(image_size),
    transforms.Grayscale(num_output_channels=1),  # Ensure images are read as grayscale
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Mean and std for single-channel images
])

train_dataset = ADNIDataset(root=data_root, split='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
