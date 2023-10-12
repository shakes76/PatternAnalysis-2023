# predict.py

import torch
import torchvision
import matplotlib.pyplot as plt

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = './OASIS'

def save_and_display_images(images, filename, nrow=8):
    """Saves and displays a grid of images."""
    # Save the image
    torchvision.utils.save_image(images, filename, nrow=nrow)

    # Display the image
    grid_img = torchvision.utils.make_grid(images, nrow=nrow)
    plt.figure(figsize=(16,8))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig(f'{filename}_plot.png', bbox_inches='tight')
    plt.close()

def generate_samples(model, test_loader, epoch):
    """Generates and saves reconstructed samples for a given epoch."""
    model.eval()  # Set model to evaluation mode
    x, _ = next(iter(test_loader))  # Get a batch of samples
    x = x[:32].to(DEVICE)

    # Reconstruct the images using the model
    x_tilde, _, _ = model(x)
    images = (torch.cat([x, x_tilde], 0).cpu().data + 1) / 2

    # Save and display the reconstructed images
    filename = f'samples3/vqvae_reconstructions_{epoch}'
    save_and_display_images(images, filename, nrow=8)

def generate_sample_from_best_model(model, test_loader, best_epoch):
    """Generates and saves a sample using the best model from a given epoch."""
    # Load the best model's weights
    model.load_state_dict(torch.load(f'samples3/checkpoint_epoch{best_epoch}_vqvae.pt'))
    model.eval()

    # Get a sample from the test set
    x, _ = next(iter(test_loader))
    x = x[:32].to(DEVICE)

    # Reconstruct the image using the model
    x_tilde, _, _ = model(x)

    # Save the reconstructed image
    filename = 'samples3/best_model_sample'
    save_and_display_images(x_tilde.cpu().data, filename)


if __name__ == "__main__":
    
