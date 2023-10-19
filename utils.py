import os
from PIL import Image
import matplotlib.pyplot as plt


def dice_coefficient(predicted, target, epsilon=1e-6):
    # Flatten the tensors
    predicted = predicted.view(-1).float()
    target = target.view(-1).float()

    intersection = (predicted * target).sum()
    dice = (2. * intersection + epsilon) / (predicted.sum() + target.sum() + epsilon)

    return dice.item()


def display_images_from_directory(directory_path, max_images=10, extensions=['.jpg', '.jpeg', '.png']):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out the files that are images with the specified extensions
    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in extensions)]

    for idx, image_file in enumerate(image_files):
        if idx >= max_images:
            break
        # Construct the full image path
        image_path = os.path.join(directory_path, image_file)

        # Open and display the image
        image = Image.open(image_path)
        image.show()


