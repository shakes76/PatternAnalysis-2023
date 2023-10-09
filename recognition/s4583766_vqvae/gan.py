"""
Generative Adversarial Network (GAN) for generating images of celebrities.

Generator: learns how to create fake data from random input and tries to make 
this produced data very similar to the real training data. Goal: produce undistinguishable
data from the real one. 

Discriminator: learns how to distinguish fake data from real data.


References: 
    - https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    - https://www.kaggle.com/code/sushant097/gan-beginner-tutorial-on-celeba-dataset/notebook
    - UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS - paper (https://arxiv.org/pdf/1511.06434.pdf)


Notes from paper:
- Replaced spatial pooling functions with strided convolutions. 
    - Allows the network to learn its own spatial downsampling. 
- Using batch normalization - stabilizes learning by normalizing the input to each unit to have zero mean and unit variance. 
- ReLU used in the generator with the exception of the output layer (which uses the Tanh function). 
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid, save_image
from tqdm.cli import tqdm

# Utility functions
def denorm(img_tensors):
    "Denormalize image tensor with specified mean and std"
    return img_tensors * stats[1][0] + stats[0][0]

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    # plt.show()

def show_batch(dl, nmax=64):
  for images, _ in dl:
    show_images(images, nmax)
    break

def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = 'generated=images-{0:0=4d}.png'.format(index)
    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)

    if show:
        fig, ax = plt.subplots(figsize=(8,8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        # plt.show()

def plot_scores(real_scores, fake_scores):
    """Plot scores from the discriminator and generator."""
    plt.close()
    plt.plot(real_scores, '-')
    plt.plot(fake_scores, '-')
    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(['Real', 'Fake'])
    plt.title('Scores');
    plt.savefig('scores.png')
    plt.close()

def plot_losses(losses_d, losses_g):
    """Plot losses from discriminator and generator."""
    plt.close()
    plt.plot(losses_d, '-')
    plt.plot(losses_g, '-')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.title('Losses');
    plt.savefig('losses.png')
    plt.close()

"""
Architecture guidelines for stable Deep Convolutional GANs
    • Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
    • Use batchnorm in both the generator and the discriminator.
    • Remove fully connected hidden layers for deeper architectures.
    • Use ReLU activation in generator for all layers except for the output, which uses Tanh.
    • Use LeakyReLU activation in the discriminator for all layers.

Note: no pooling layers are used. 
"""
class Discriminator(nn.Module):
    """
    Detect fake images from real images (encoder).

    Takes a 3 x 64 x 64 image, and outputs a single number representing 
    probability of it being real or fake. 
    """
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        # Input: N x channels_img x 64 x 64
        self.net = nn.Sequential(
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1), # With a stride of 2, need padding of 1 - prevents downsampling.
            nn.LeakyReLU(0.2), # Allows 0.2 of the negative
            self._block(3, 64, 2),
            self._block(64, 128, 2),
            self._block(128, 256, 2),
            self._block(256, 512, 2), 
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid() # convert to probability output [0,1]
        )

    # Create a discriminator block with a convolutional layer, batch normalization, and leaky ReLU activation.
    def _block(self, in_planes, planes, stride):
        # Use of strided convolution is better than downsampling - model pools itself. 
        return nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
# Create the generator model.
class Generator(nn.Module):
    """ 
    Produce fake images sampled from the latent space (decoder).

    Needs to receive latent space vector as an input, and map to data space (image). 
        - Hence, need to create an image that's the same size as training images (3x64x64).
    Batch norm after the conv-transpose layers helps with vanishing gradient problem.
        - normalizing input to have zero mean and unit variance = deals with poor initialization. 
    """
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        # Input: N x channels_noise x 1 x 1
        self.net = nn.Sequential(
            self._block(channels_noise, 512, 1), 
            self._block(512, 256, 2), 
            self._block(256, 128, 2), 
            self._block(128, 64, 2), 
            nn.ConvTranspose2d(64, channels_img, kernel_size=4, stride=2, padding=1), # N x channels_img x 64 x 64
            nn.Tanh() # convert to [-1, 1] 
        )

    # Create a generator block with a transposed convolutional layer, batch normalization, and ReLU activation.
    def _block(self, in_plane, plane, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_plane, plane, kernel_size=4, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(plane),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)

"""
Train the discriminator. 

First check the real images, then the fake ones. 
"""
def train_discriminator(real_images, opt_d):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Pass real images through  discriminator, so discriminator can 
    # learn what they look like. 
    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = criterion(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    # Generate fake images. Latent space = representation of compressed data.
    # Use the same seed every time, so we can compare the generated images.
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    # Pass Fake images through discriminator
    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = criterion(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    """
    Add the real and fake loss together and backpropagate the combined loss.
    # real = D(x), fake = (1-D(G(z))).

    D(G(z)) is the prob that the output of G is real image. 
    D(x) is the prob that the input to D is real image.

    We want to minimize loss of D(x), and maximize loss of D(G(z)), 
        so: loss = log(D(x)) - log(1-D(G(z)))
    """
    loss = real_loss + fake_loss
    # Backpropagate loss, and update weights.
    loss.backward()
    opt_d.step()
    return loss.item(), real_score, fake_score

"""
Train the generator by generating fake images that attempt to fool 
the discriminator.

Taking the image and encoding it into noise, and the noise is random Gauss noise. 
    - Then we sample from Gauss distribution, to then decode it into an image.
"""
def train_generator(opt_g):
    # Clear generator gradients
    opt_g.zero_grad()

    # Generate a batch of fake images. 
    latent = torch.randn(batch_size, latent_size, 1,1, device=device)
    fake_images = generator(latent)

    # Try to fool the discriminator
    preds = discriminator(fake_images)
    # Set labels to '1' to indicate they're fake. 
    targets = torch.ones(batch_size, 1, device=device)
    # Minimize loss of D(G(z)). 
    loss = criterion(preds, targets)

    # Backpropagate loss, and update weights.
    loss.backward()
    opt_g.step()

    return loss.item()

def fit(epochs, lr, start_idx = 1):
    torch.cuda.empty_cache()

    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    # Create optimizers for generator and discriminator using Adam. Adam analyzes historical gradients, to adjust the learning rate for each parameter in real-time, resulting in faster convergence and better performance. 
    # Adam is a combination of RMSProp + Momentum, it uses moving averages of parameters. 
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_loader):
            real_images = real_images.to(device)
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
        # Save generated images
        save_samples(epoch+start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Setup file paths TODO: update these
path = os.getcwd() + '/'
data_path = path + 'data/celeba/'
sample_dir = 'generated/'

# Hyper-parameters
image_size = 64
batch_size = 128
latent_size = 100
dataset_size = 100000

# Normalize to -1 to 1
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)  # mean, std

"""
Transform the training data:
1. Resize the image to 64 x 64
2. Crop the image to 64 x 64 (pick central square crop of it)
3. Convert the image to a PyTorch tensor
4. Normalize the image (mean = 0.5, std = 0.5) so that the image has values between -1 to 1.
"""
transform_train = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size), # Pick central square crop of it
    transforms.ToTensor(), 
    transforms.Normalize(*stats) 
])

"""
Use ImageFolder to load the images from the generated/ directory, instead of using the 
torchvision.datasets.CelebA dataset. This was due to issues downloading the entire dataset via 
the torchvision.datasets.CelebA method. 
"""
trainset = ImageFolder(root=data_path, transform=transform_train)
# Select a random 100,000 image subset from the dataset.
train_ds = torch.utils.data.Subset(trainset, np.random.choice(len(trainset), dataset_size, replace=False))

"""
Create a dataloader to load the images in batches of size batch_size. We'll iterate over this dataloader 
during training.

Shuffle = True to shuffle the images in each epoch.
Pin_memory = True to speed up the data transfer to GPU.
batch_size = 128 to use 128 images + features in each iteration.
"""
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=1)

# Use Binary Cross Entropy Loss: because we're trying to classify as one or the other. 
# (otherwise could use categorical cross entropy loss).
criterion = nn.BCELoss()
   
discriminator = Discriminator(channels_img=3, features_d=64)
generator = Generator(channels_noise=latent_size, channels_img=3, features_g=64)

# Move the generator and discriminators to the device
generator = generator.to(device)
discriminator = discriminator.to(device)

# Save the first image checkpoint before training begins. 
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
save_samples(0, fixed_latent)

lr = 0.00025
epochs = 5
history = fit(epochs, lr)

# Uncomment to save the model checkpoints
# torch.save(generator.state_dict(), 'G.pth')
# torch.save(discriminator.state_dict(), 'D.pth')

# Plot losses and scores
losses_g, losses_d, real_scores, fake_scores = history

plot_losses(losses_d, losses_g)
plot_scores(real_scores, fake_scores)

