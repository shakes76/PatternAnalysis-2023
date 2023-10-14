"""
Hugo Burton
s4698512
20/09/2023

predict.py
implements usage of *trained* model.
Prints out results and provides visualisations
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import ADNI, OASIS
from modules import VectorQuantisedVAE
from train import generate_samples


def predict(dataset: str, device: torch.device, model_name: str = "best.pt"):
    """
    Takes a trained model of a dataset and generates (fake) samples.
    Stores them in a folder and also displays a sample of the samples.

    Args:
        dataset (str): OASIS or ADNI dataset
        device (torch.device): The device used to generate the samples on
        model (str): name of the model save file used to generate samples
    """

    # Hyperparameters. These can be arbitrary as it's actually read from the save file
    num_channels = 1

    # Input image resolution
    if dataset == "OASIS":
        image_x, image_y = 256, 256
    elif dataset == "ADNI":
        image_x, image_y = 240, 256

    # Size/dimensions within latent space
    hidden_dim = 128     # Number of neurons in each layer
    K = 32      # Size of the codebook

    # Create the generated samples directory if it doesn't exist
    # This will also store the loss and ssim plots
    output_dir = os.path.join(
        "..", "PatternAnalysisData", "generated", dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Define model.
    model = VectorQuantisedVAE(input_channels=num_channels,
                               output_channels=num_channels, hidden_channels=hidden_dim, num_embeddings=K)
    # Load the saved model
    model_path = os.path.join(
        "..", "PatternAnalysisData", "ModelParams", model_name)
    print(model_path)
    model_loaded = torch.load(model_path)

    model.load_state_dict(model_loaded['model_state_dict'])

    # Load SSIM, Train and Validation Losses
    ssim_values = model_loaded['ssim_values']
    train_losses = model_loaded['train_losses']
    validation_losses = model_loaded['validation_losses']

    # Plot SSIM values over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(ssim_values) + 1),
             ssim_values, marker='o', linestyle='-')
    plt.title('SSIM Progress')
    plt.xlabel('Epoch')
    plt.ylabel('SSIM Value')
    plt.grid(True)
    plt.savefig(f'{output_dir}/ssim_plot.png')
    plt.show()

    # Plot both training and validation losses in the same plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses,
             label='Train Loss', marker='o', linestyle='-')
    plt.plot(range(1, len(validation_losses) + 1), validation_losses,
             label='Validation Loss', marker='o', linestyle='-')
    plt.title('Loss Progress')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'{output_dir}/loss_plot.png')
    plt.show()

    # Move model to CUDA GPU
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()

    data_path = os.path.join("..", "PatternAnalysisData", "datasets", dataset)

    # Define a transformation to apply to the images (e.g., resizing)
    transform = transforms.Compose([
        transforms.Resize((image_x, image_y)),  # Adjust the size as needed
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    if dataset == "OASIS":
        data_object = OASIS(data_path, copy=False, transform=transform)
    elif dataset == "ADNI":
        data_object = ADNI(data_path, copy=False,
                           transform=transform, sampleType="AD")

    # Define a data loader for generating samples
    sample_loader = data_object.test_loader

    # Get a batch of data for generating samples
    sample_images, _ = next(iter(sample_loader))

    # Generate samples
    with torch.no_grad():
        # Replace your_fixed_images with actual data
        fake_images, encoded_images, latent_images = generate_samples(
            sample_images, model, device)

    # Generate and save each sample
    for i in range(len(fake_images)):
        sample = sample_images[i].cpu().numpy().squeeze()
        sample_filename = os.path.join(output_dir, f'sample_{i + 1:03d}.png')
        plt.imsave(sample_filename, sample, cmap='gray')

    # Generate and save each sample
    for i in range(len(fake_images)):
        sample = fake_images[i].cpu().numpy().squeeze()
        sample_filename = os.path.join(output_dir, f'fake_{i + 1:03d}.png')
        plt.imsave(sample_filename, sample, cmap='gray')

    # Generate and save each encoded image
    for i in range(len(encoded_images)):
        sample = encoded_images[i].cpu().numpy().squeeze()
        # Add a single channel dimension
        sample = np.expand_dims(sample, axis=-1)
        sample_filename = os.path.join(output_dir, f'encoded_{i + 1:03d}.png')
        plt.imsave(sample_filename, sample, cmap='jet')

    # Visualize and save the generated samples
    fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        ax.imshow(fake_images[i].cpu().numpy().squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'samples.png'))  # Save the figure

    # Display the generated samples
    plt.show()


def main():
    # Dataset selection
    dataset = "OASIS"

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the model to use the best one trained
    model = "best_model.pt"

    predict(dataset, device, model)


if __name__ == "__main__":
    main()
