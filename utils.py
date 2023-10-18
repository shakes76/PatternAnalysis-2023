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


def display_images_from_directory(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter out the files that are JPEG images
    jpg_files = [f for f in files if f.lower().endswith('.jpg')]

    for jpg_file in jpg_files:
        # Construct the full image path
        image_path = os.path.join(directory_path, jpg_file)

        # Open and display the image
        image = Image.open(image_path)
        image.show()

