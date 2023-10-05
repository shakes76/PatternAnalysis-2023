"""
Hugo Burton
s4698512
20/09/2023

train.py
Contains source code for training, validating, testing and saving model.
Model imported from modules.py and data loader imported from dataset.py
Plots losses and metrics during training.
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets        # Will test on MNIST first
from torchvision.utils import save_image
from tqdm import tqdm
import os

from modules import VectorQuantisedVAE
from dataset import OASIS, ADNI


def train(train_loader: DataLoader, model: VectorQuantisedVAE, optimiser: torch.optim.Adam, device: torch.device, beta: int):
    """
    Inside of the training loop for the VQ-VAE
    Args:
        train_loader (DataLoader): the loader for the trainind data
        model (VectorQuantisedVAE): the VQ-VAE Model
        optimiser (torch.optim.Adam): The learning rate optimiser
        device (torch.device): the device that the training runs on
        beta (int): loss weight
    """

    # Loop over the images in the data loader
    for images, _ in train_loader:
        # Move the images in the batch to the GPU
        images = images.to(device)

        optimiser.zero_grad()

        x_til, z_e_x, z_q_x = model(images)

        # Reconstruction Loss
        recon_loss = F.mse_loss(x_til, images)

        # Vector Quantised Objective Function
        # We need to detach the gradient becuse gradients won't work in disctete space for backpropagation
        # vq_loss = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach())

        fin_loss = recon_loss + beta * commit_loss
        fin_loss.backward()

        optimiser.step()


def test(test_loader: DataLoader, model: VectorQuantisedVAE, device: torch.device):
    """
    Tests the model with testing set

    """
    with torch.no_grad():
        # Initialise losses to 0
        recon_loss, vq_loss = 0, 0

        # Loop over images in test_loader
        for images, _ in test_loader:
            # Move images to device
            images = images.to(device)

            # Generate samples
            x_til, z_e_x, z_q_x = model(images)

            # Add to loss using mean squared error
            recon_loss += F.mse_loss(x_til, images)
            vq_loss += F.mse_loss(z_q_x, z_e_x)

        recon_loss /= len(test_loader)
        vq_loss /= len(test_loader)

        return recon_loss.item(), vq_loss.item()


def generate_samples(images, model, device):
    """
    Generates image samples using model without any noise. Best for training
    """

    with torch.no_grad():
        images = images.to(device)

        # Invoke forward pass on model
        x_tilde, _, _ = model(images)

    return x_tilde


def main():
    # Some preliminary stuff

    # GPU Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = "ADNI"   # Choose dataset ADNI, OASIS

    # Define Global hyper parameters (for both datasets)

    # Learning rate
    learning_rate = 1.2e-4

    # Number of epochs during training
    num_epochs = 160

    # Trade off between reconstruction loss and KL Divergence loss
    # Reconstruction loss measures how similar of an output the model can produce from the input
    # data (when passed through the latent codebook). KL Divergence loss controls the latent space
    # of the VQ-VAE by forcing the encoder to follow a regularlised normal distribution in order to
    # prevent overfitting. A higher beta value (closer to 1) lowers KL Divergence Loss while a
    # lower beta value (closer to 0) lowers reconstruction loss.
    beta = 0.4

    # Greyscale data: 1 channel
    num_channels = 1

    dataset_path = os.path.join(".", "datasets", dataset)

    if dataset == "OASIS":
        # Input image dimensions
        image_x, image_y = 256, 256

        # Number of neurons in each (inner/hidden) layer of the neural network
        hidden_dim = 32

        # Size/dimensions within latent space
        K = 32      # Size of the codebook
        # Dimensions of each vector within latent space

        beta = 0.75

        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            # Images already 256x256 but can't hurt to ensure size is correct
            transforms.Resize((image_x, image_y)),

            # Convert images to single channel greyscale
            transforms.Grayscale(num_output_channels=num_channels),

            # Finally convert images into Tensor
            transforms.ToTensor(),
        ])

        # Define data loader object
        oasis = OASIS(dataset_path, transform=transform, copy=False)

        # Obtain data loaders from oasis object
        train_loader = oasis.train_loader
        test_loader = oasis.test_loader
        validate_loader = oasis.validate_loader

    elif dataset == "ADNI":
        # Input image dimensions
        image_x, image_y = 240, 256

        # Number of neurons in each (inner/hidden) layer of the neural network
        hidden_dim = 32

        # Size/dimensions within latent space
        K = 32      # Size of the codebook
        # Dimensions of each vector within latent space

        # beta Trade off between reconstruction loss and commitment loss
        beta = 0.75

        # Choose what type of samples to generate
        sampleType = "AD"     # AD = generate Alzheimer's samples; NC = generate control samples

        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            # Images already 256x256 but can't hurt to ensure size is correct
            transforms.Resize((image_x, image_y)),

            # Convert images to single channel greyscale
            transforms.Grayscale(num_output_channels=num_channels),

            # Finally convert images into Tensor
            transforms.ToTensor(),
        ])

        # Define data loader object
        adni = ADNI(dataset_path, sampleType=sampleType,
                    transform=transform, copy=False)

        # Obtain data loaders from adni object
        train_loader = adni.train_loader
        test_loader = adni.test_loader
        validate_loader = adni.test_loader

    # Dataset Imported
    print(f"{dataset} Dataset Imported :)\n------\n")

    # Define Model and move to GPU
    model = VectorQuantisedVAE(input_channels=num_channels,
                               output_channels=num_channels, hidden_channels=hidden_dim, num_embeddings=K)
    model = model.to(device)
    print("Model defined", model)

    # Optimiser as Adam
    print("Initialising optimiser")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Optimiser initialised", optimiser)

    # Train model
    print("Retrieving first batch of images from test loader")
    fixed_images, _ = next(iter(test_loader))
    print(f"{len(fixed_images)} images loaded")

    # Initialise Directory to save model to
    save_filename = os.path.join(".", "ModelParams")
    if not os.path.exists(save_filename):
        os.makedirs(save_filename)

    # Keep track of best loss so far and initialise to arbitrary -1
    best_loss = -1.

    for epoch in range(num_epochs):
        train(train_loader, model, optimiser, device, beta)
        loss, _ = test(validate_loader, model, device)

        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {loss:.4f}")

        # Save the model if it has the lowest validation loss so far
        if (epoch == 0) or (loss < best_loss):
            best_loss = loss
            with open(f'{save_filename}/best.pt', 'wb') as f:
                torch.save(model.state_dict(), f)

        # Save the model at each epoch
        with open(f'{save_filename}/model_{epoch + 1}.pt', 'wb') as f:
            torch.save(model.state_dict(), f)


def inference(digit, model, dataset):
    """
    Generates (num_examples) of a particular digit.
    Specifically we extract an example of each digit,
    then after we have the mu, sigma representation for
    each digit we can sample from that.

    After we sample we can run the decoder part of the VAE
    and generate examples.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    images = []
    idx = 0
    for x, y in dataset:
        if y == idx:
            images.append(x)
            idx += 1
        if idx == 10:
            break

    encodings_digit = []
    for d in range(10):
        with torch.no_grad():
            mu, sigma = model.encode(images[d].view(1, 784).to(device))
        encodings_digit.append((mu, sigma))

    mu, sigma = encodings_digit[digit]
    for example in range(5):
        epsilon = torch.randn_like(sigma)
        z = mu + sigma * epsilon
        out = model.decode(z)
        out = out.view(-1, 1, 28, 28)
        save_image(out, os.path.join(
            "mnistout", f"generated_{digit}_ex{example}.png"))


if __name__ == "__main__":
    main()
