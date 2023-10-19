"""
utils.py
Utilities for visual transformer.

Author: Atharva Gupta
Date Created: 17-10-2023
"""
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from datasplit import load_data

def plot_image(dataset):
    """
    Plots an image from the given dataset.
    """
    plt.figure(figsize=(5, 5))
    dataset_it = iter(dataset)
    image = next(dataset_it)[0][0]
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    train, val, test = load_data()
    plot_image(train)

