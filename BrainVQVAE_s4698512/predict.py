"""
Hugo Burton
s4698512
20/09/2023

predict.py
implements usage of *trained* model.
Prints out results and provides visualisations
"""

import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset import ADNI, OASIS
from modules import VectorQuantisedVAE
from train import generate_samples


def predict(dataset: str, num_samples: int, device: torch.device, model_name: str = "best.pt"):
    """
    Takes a trained model of a dataset and generates (fake) samples.
    Stores them in a folder and also displays a sample of the samples.

    Args:
        dataset (str): OASIS or ADNI dataset
        num_samples (int): Number of samples to generate
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
    K = 512      # Size of the codebook

    # Peturbance
    noise_scale = 0.001

    # Define model.
    model = VectorQuantisedVAE(input_channels=num_channels,
                               output_channels=num_channels, hidden_channels=hidden_dim, num_embeddings=K)
    # Load the saved model
    model_path = os.path.join(".", "ModelParams", model_name)
    print(model_path)
    model.load_state_dict(torch.load(model_path)
                          )  # Update the path accordingly
    # Move model to CUDA GPU
    model = model.to(device)
    # Set the model to evaluation mode
    model.eval()

    data_path = os.path.join(".", "datasets", dataset)

    # Define a transformation to apply to the images (e.g., resizing)
    transform = transforms.Compose([
        transforms.Resize((image_x, image_y)),  # Adjust the size as needed
        transforms.Grayscale(num_output_channels=num_channels),
        transforms.ToTensor(),
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

    def generate_peturbed_samples(images, model, device, noise_scale=0.1):
        with torch.no_grad():
            images = images.to(device)

            # Invoke forward pass on model
            x_tilde, _, _ = model.forward_peturb(images, noise_scale)

        return x_tilde

    # Generate samples
    with torch.no_grad():
        # Replace your_fixed_images with actual data
        fake_images = generate_samples(
            sample_images, model, device)

    # Create the generated samples directory if it doesn't exist
    output_dir = f"./{dataset}_generated"
    os.makedirs(output_dir, exist_ok=True)

    # Generate and save each sample
    for i in range(len(fake_images)):
        sample = fake_images[i].cpu().numpy().squeeze()
        sample_filename = os.path.join(output_dir, f'fake_{i + 1:03d}.png')
        plt.imsave(sample_filename, sample, cmap='gray')

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

    # Number of samples to generate
    num_samples = 128

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set the model to use the best one trained
    model = "model_60.pt"

    predict(dataset, num_samples, device, model)


if __name__ == "__main__":
    main()


# Yet to implement structured similarlity metric

# from skimage.metrics import structural_similarity as ssim
# from skimage import io

# original_image = io.imread('original_image.png')
# reconstructed_image = io.imread('reconstructed_image.png')

# # Calculate SSIM
# ssim_score = ssim(original_image, reconstructed_image)

# print(f"SSIM Score: {ssim_score}")