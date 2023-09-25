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
from torchvision.utils import save_image, make_grid
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


from modules import VectorQuantizedVAE
from dataset import OASIS, ADNI


def train(train_loader: DataLoader, model: VectorQuantizedVAE, optimiser: torch.optim.Adam, device: torch.device, beta: int):
    """
    Inside of the training loop for the VQ-VAE
    Args:
        train_loader (DataLoader): the loader for the trainind data
        model (VectorQuantizedVAE): the VQ-VAE Model
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
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach())

        fin_loss = recon_loss + beta * commit_loss
        fin_loss.backward()

        optimiser.step()
    print("1 Training loop done")


def test(test_loader: DataLoader, model: VectorQuantizedVAE, device: torch.device):
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
    print(f"Generating samples on {device}")
    with torch.no_grad():
        print(f"Passing images to {device}")
        images = images.to(device)
        print(f"Images on {device}")

        # Invoke forward pass on model
        x_tilde, _, _ = model(images)

    return x_tilde


def main():
    # Some preliminary stuff

    # GPU Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = "OASIS"   # ADNI, OASIS

    # Hyper parameters
    learning_rate = 3e-4
    num_epochs = 10
    beta = 1.0

    dataset_path = os.path.join(".", "datasets", dataset)

    if dataset == "OASIS":
        # Load OASIS Dataset
        # Greyscale data. Observe below in the transform I set num_output_channels = num_channels
        num_channels = 1
        image_x, image_y = 256, 256
        # Size/dimensions within latent space
        hidden_dim = 32     # Number of neurons in each layer
        K = 32      # Size of the codebook
        # Dimensions of each vector within latent space
        z_dim = 8

        oasis_data_path = os.path.join(".", "datasets", "OASIS")

        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            transforms.Resize((image_x, image_y)),  # Adjust the size as needed
            transforms.Grayscale(num_output_channels=num_channels),
            transforms.ToTensor(),
        ])

        oasis = OASIS(oasis_data_path, transform=transform)

        train_loader = oasis.train_loader
        test_loader = oasis.test_loader
        validate_loader = oasis.validate_loader

    elif dataset == "ADNI":
        # Load ADNI dataset
        num_channels = 1

        pass
    elif dataset == "MNIST":
        input_dim = 28 * 28
        # Hyper parameters

        # Size/dimensions within latent space
        hidden_dim = 200

        # Dimensions of each vector within latent space
        z_dim = 20

        # epoch count
        num_epochs = 20

        # batch size
        batch_size = 64

        # learning rate
        learning_rate = 3e-4    # Karpathy constant

        # Load Dataset and normalise it to pixel values between 0 and 1. Divide by 255
        dataset = datasets.MNIST(
            root=os.path.join(".", "datasets", "MNIST"), train=True, download=False, transform=transforms.ToTensor())

        train_loader = DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=True)

    # Dataset Imported
    print(f"{dataset} Dataset Imported :)\n------\n")

    # Define Model and move to GPU
    model = VectorQuantizedVAE(input_channels=num_channels,
                               output_channels=num_channels, hidden_channels=hidden_dim, num_embeddings=K)
    model = model.to(device)
    print("Model defined", model)

    # Optimiser as Adam
    print("Initialising optimiser")
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    print("Optimiser initialised", optimiser)

    # Reconstruction Loss function. Use BCELoss
    print("initialising Loss function")
    loss_fn = torch.nn.MSELoss(reduction="sum")
    print("MSELoss initialised")

    # Train model
    print("Retrieving first batch of images from test loader")
    fixed_images, _ = next(iter(test_loader))
    print(f"{len(fixed_images)} images loaded")
    fixed_grid = make_grid(fixed_images, nrow=8, range=(-1, 1), normalize=True)
    print("Fixed Grid computed")

    # Generate the samples first once
    reconstruction = generate_samples(fixed_images, model, device)
    grid = make_grid(reconstruction.cpu(), nrow=8,
                     range=(-1, 1), normalize=True)

    print("Reconstruction Done")

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

        # Generate and display samples
        with torch.no_grad():
            reconstruction = generate_samples(fixed_images, model, device)

        # Save the model at each epoch
        with open(f'{save_filename}/model_{epoch + 1}.pt', 'wb') as f:
            torch.save(model.state_dict(), f)

    # Visualize the generated samples using matplotlib
    fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(reconstruction[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.show()


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
