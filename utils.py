import os
from PIL import Image
import matplotlib.pyplot as plt


def dice_coefficient(predicted, target, epsilon=1e-6):
    """
    Calculate the Dice Coefficient for two tensors.

    Args:
        predicted (tensor): Predicted tensor.
        target (tensor): Ground truth tensor.
        epsilon (float): Small constant for numerical stability.

    Returns:
        float: Dice Coefficient.
    """
    # Flatten the tensors
    predicted = predicted.view(-1).float()
    target = target.view(-1).float()

    intersection = (predicted * target).sum()
    dice = (2. * intersection + epsilon) / (predicted.sum() + target.sum() + epsilon)

    return dice.item()


def display_images_from_directory(directory_path, max_images=10, extensions=['.jpg', '.jpeg', '.png']):
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory {directory_path} does not exist.")
    
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out the files that are images with the specified extensions
    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in extensions)]

    if not image_files:
        raise ValueError(f"No image files found in the directory {directory_path}.")

    for idx, image_file in enumerate(image_files):
        if idx >= max_images:
            break
        # Construct the full image path
        image_path = os.path.join(directory_path, image_file)

        # Open and display the image
        try:
            image = Image.open(image_path)
            image.show()
        except Exception as e:
            print(f"Error opening {image_path}: {e}")



