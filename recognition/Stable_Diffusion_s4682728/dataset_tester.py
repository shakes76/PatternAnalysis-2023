import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import * 

if __name__ == '__main__':
    image_dir = os.path.expanduser('/Users/Eiji/Documents/ML/Data/OASIS_Brain/keras_png_slices_train')

    # Check if Python can see the path
    if os.path.exists(image_dir):
        print(f"Directory exists: {image_dir}")
    else:
        print(f"Directory does not exist: {image_dir}")
        print(f"Current working directory: {os.getcwd()}")

    # Initialize DataLoader
    data_loader = prepare_dataset(batch_size=4)

    # Fetch and visualize a batch of images
    batch = next(iter(data_loader))
    for i, image in enumerate(batch):
        plt.subplot(1, 4, i+1)
        plt.imshow(image.squeeze(0))
        plt.axis('off')
    plt.show()