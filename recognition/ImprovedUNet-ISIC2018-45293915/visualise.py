import tensorflow as tf
from dataset import SkinLesionDataset
from matplotlib import pyplot as plt

# Example usage:
dataset = SkinLesionDataset(data_dir="datasets")
train_data = dataset.train_dataset

# Define the number of images to show
n_images = 5

# Get the first n_images pairs from the training dataset
image_mask_pairs = [item for item in train_data.take(n_images)]

# Plot images and masks
fig, axes = plt.subplots(nrows=2, ncols=n_images, figsize=(15, 15))
for i in range(n_images):
    image, mask = image_mask_pairs[i]
    axes[0, i].imshow(image)
    axes[0, i].axis("off")
    axes[0, i].set_title("Image")
    
    axes[1, i].imshow(mask, cmap='gray')  # Assuming grayscale masks
    axes[1, i].axis("off")
    axes[1, i].set_title("Mask")

plt.show()
