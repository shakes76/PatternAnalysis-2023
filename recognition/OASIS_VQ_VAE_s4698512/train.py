"""
Hugo Burton
s4698512
20/09/2023

train.py
Contains source code for training, validating, testing and saving model.
Model imported from modules.py and data loader imported from dataset.py
Plots losses and metrics during training.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

from modules import VectorQuantisedVAE
from dataset import OASIS, ADNI

DATASETS_PATH = os.path.join("..", "PatternAnalysisData", "datasets")


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

    ssim_list = []
    recon_losses = []

    # Loop over the images in the data loader
    for images, _ in train_loader:
        # Move the images in the batch to the GPU
        images = images.to(device)

        optimiser.zero_grad()

        x_til, z_e_x, z_q_x = model(images)

        # Reconstruction Loss
        recon_loss = F.mse_loss(x_til, images)
        recon_losses.append(recon_loss.item())

        # Compute SSIM
        images_cpu = images.to('cpu').detach().numpy()
        x_til_cpu = x_til.to('cpu').detach().numpy()

        batch_ssim = ssim(
            images_cpu[0, 0], x_til_cpu[0, 0], data_range=images_cpu[0, 0].max() - images_cpu[0, 0].min())

        ssim_list.append(batch_ssim)

        # Vector Quantised Objective Function
        # We need to detach the gradient becuse gradients won't work in disctete space for backpropagation
        vq_loss = F.mse_loss(z_q_x, z_e_x.detach())

        # Commitment objective
        commit_loss = F.mse_loss(z_e_x, z_q_x.detach())

        fin_loss = recon_loss + vq_loss + beta * commit_loss
        fin_loss.backward()

        optimiser.step()

    return sum(ssim_list) / len(ssim_list), sum(recon_losses)/len(recon_losses)


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
        x_tilde, z_e_x, z_q_x = model(images)

    return x_tilde, z_e_x, z_q_x


def main():
    # Some preliminary stuff

    # GPU Device Config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = "OASIS"   # Choose dataset ADNI, OASIS

    # Define Global hyper parameters (for both datasets)

    # Learning rate
    learning_rate = 2e-4

    # Number of epochs during training
    num_epochs = 40

    # Trade off between reconstruction loss and KL Divergence loss
    # Reconstruction loss measures how similar of an output the model can produce from the input
    # data (when passed through the latent codebook). KL Divergence loss controls the latent space
    # of the VQ-VAE by forcing the encoder to follow a regularlised normal distribution in order to
    # prevent overfitting. A higher beta value (closer to 1) lowers KL Divergence Loss while a
    # lower beta value (closer to 0) lowers reconstruction loss.
    beta = 0.25

    # Greyscale data: 1 channel
    num_channels = 1

    # Number of embeddings within latent space's codebook. Each embedding is a discrete point within the
    # latent space that an input data point can be snapped to (closest point). A higher number of
    # embeddings results in a finer latent space meaning more information can be retained from input
    # samples. However, too high may result in overfitting.
    K = 32

    # Number of neurons in each (inner/hidden) layer of the neural network
    hidden_dim = 128

    dataset_path = os.path.join(DATASETS_PATH, dataset)

    if dataset == "OASIS":
        # Input image dimensions
        image_x, image_y = 256, 256

        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            # Images already 256x256 but can't hurt to ensure size is correct
            transforms.Resize((image_x, image_y)),

            # Convert images to single channel grayscale
            transforms.Grayscale(num_output_channels=num_channels),

            # Finally, convert images into Tensor
            transforms.ToTensor(),
            # Normalise images to range [-1, 1]
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Check if "OASIS_processed" folder does not exist
        oasis_processed_folder = os.path.join(DATASETS_PATH, "OASIS_processed")
        copy = not os.path.exists(oasis_processed_folder)

        # Define data loader object
        oasis = OASIS(dataset_path, transform=transform, copy=copy)

        # Obtain data loaders from oasis object
        train_loader = oasis.train_loader
        test_loader = oasis.test_loader
        validate_loader = oasis.validate_loader

    elif dataset == "ADNI":
        # Input image dimensions
        image_x, image_y = 240, 256

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
            # Normalise images to range [-1, 1]
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Check if "ADNI_processed" folder does not exist
        adni_processed_folder = os.path.join(DATASETS_PATH, "ADNI_processed")
        copy = not os.path.exists(adni_processed_folder)

        # Define data loader object
        adni = ADNI(dataset_path, sampleType=sampleType,
                    transform=transform, copy=copy)

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
    save_filename = os.path.join("..", "PatternAnalysisData", "ModelParams")
    if not os.path.exists(save_filename):
        os.makedirs(save_filename)

    # Keep track of best loss so far and initialise to arbitrary -1
    best_loss = 999.

    # Initialise empty arrays to store metrics
    ssim_values = []
    train_losses = []
    validation_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Progress"):
        # Train and return structured similarity metric and train loss
        ssim, train_loss = train(train_loader, model, optimiser, device, beta)
        ssim_values.append(ssim)
        train_losses.append(train_loss)

        # Validation
        validation_loss, _ = test(validate_loader, model, device)
        validation_losses.append(validation_loss)

        # Output to console
        tqdm.write(
            f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} | Validation Loss: {validation_loss:.4f} | SSIM Value: {ssim:.4f}")

        # Save the model and metrics in a checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'ssim_values': ssim_values,
            'train_losses': train_losses,
            'validation_losses': validation_losses
        }

        with open(f'{save_filename}/model_{epoch + 1}.pt', 'wb') as f:
            torch.save(checkpoint, f)

        # Save the model if it has the lowest validation loss so far
        if validation_loss < best_loss:
            best_loss = validation_loss
            with open(f'{save_filename}/best.pt', 'wb') as f:
                torch.save(checkpoint, f)


if __name__ == "__main__":
    main()
