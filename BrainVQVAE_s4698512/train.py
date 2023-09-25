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


from modules import VectorQuantizedVAE
from dataset import OASIS, ADNI


def train(train_loader, model, optimiser, device, beta, steps):
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
        steps += 1


def test(test_loader, model, device, steps):
    """
    Tests the model with testing set

    """


def generate_samples(images, model, device):
    with torch.no_grad():
        images = images.to(device)
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
        num_channels = 1

        oasis_data_path = os.path.join(".", "datasets", "OASIS")
        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust the size as needed
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
    model = VectorQuantizedVAE(input_dim, hidden_dim, z_dim)
    model = model.to(device)

    # Optimiser as Adam
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Reconstruction Loss function. Use BCELoss
    loss_fn = torch.nn.MSELoss(reduction="sum")

    # Train model
    steps = 0
    for epoch in range(num_epochs):
        # use tqdm loop for progress bar
        loop = tqdm(enumerate(train_loader))
        for i, (x, y) in loop:
            # Forward pass
            # Move x to GPU
            x = x.to(device).view(x.shape[0], input_dim)

            x_recon, mu, sigma = model(x)

            # Compute Loss

            recon_loss = loss_fn(x_recon, x)

            kl_divergence = - \
                torch.sum(1 + torch.log(sigma.pow(2)) -
                          mu.pow(2) - sigma.pow(2))

            # Back propagation
            loss = recon_loss + kl_divergence
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            steps += 1
            loop.set_postfix(loss=loss.item())

    # call inference

    for idx in range(10):
        inference(idx, model, dataset)
    # # Train
    # train(train_loader, model, optimiser, device, beta, steps)

    # # Perform Validation (mid epoch)
    # loss, _ = test(validation_loader, model, device, steps)

    # reconstruction = generate_samples(fixed_images, model, device)
    # grid = make_grid(reconstruction.cpu(), nrow=8,
    #                  range=(-1, 1), normalize=True)

    # if (epoch == 0) or (loss < best_loss):
    #     best_loss = loss
    #     with open('{0}/best.pt'.format(save_filename), 'wb') as f:
    #         torch.save(model.state_dict(), f)
    # with open('{0}/model_{1}.pt'.format(save_filename, epoch + 1), 'wb') as f:
    #     torch.save(model.state_dict(), f)


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
