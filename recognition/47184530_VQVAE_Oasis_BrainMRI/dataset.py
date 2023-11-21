import os
import zipfile
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from prettytable import PrettyTable

# Ensure that PyTorch uses the GPU (if available) or CPU otherwise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mounting Google Drive to access files. Note: This is specific to Google Colab.
drive.mount('/content/drive')

# Define the directory where the output will be saved
OUTPUT_DIR = "/content/drive/MyDrive/Colab_Notebooks_Course/image_process/A3/OUTPUT2"

# Create the directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Dataset class to handle brain slice images
class BrainSlicesDataset(Dataset):
    def __init__(self, image_slices):
        self.image_slices = image_slices

    def __len__(self):
        # Return the total number of image slices
        return len(self.image_slices)

    def __getitem__(self, idx):
        image = self.image_slices[idx]

        # Ensure the image has a channel dimension (grayscale images may not have one)
        if len(image.shape) == 2:  # If the image is of shape [Height, Width]
            image = torch.unsqueeze(image, 0)  # Convert it to [1, Height, Width]

        return image


# Function to load and extract image slices from a zip file
def get_image_slices():
    # Path to the zipped dataset
    zip_path = "/content/drive/MyDrive/Colab_Notebooks_Course/image_process/A3/testgans/GAN_Dataset.zip"
    extraction_path = "/content/GAN_Dataset"
    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_path)

    # Define the directories for training, testing, and validation datasets
    parent_dir = "/content/GAN_Dataset"
    train_path = os.path.join(parent_dir, "keras_png_slices_train")
    test_path = os.path.join(parent_dir, "keras_png_slices_test")
    val_path = os.path.join(parent_dir, "keras_png_slices_validate")

    # Helper function to load images from a directory
    def load_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            # Open the image, convert to grayscale, and resize to 128x128 pixels
            img = Image.open(os.path.join(folder_path, filename)).convert('L').resize((128, 128))
            if img is not None:
                # Convert the image to a tensor and append to the list
                images.append(torch.tensor(np.array(img, dtype=np.float32)))
        return torch.stack(images)  # Convert list of tensors to a single tensor

    # Load images from each directory
    train_images = load_images_from_folder(train_path)
    test_images = load_images_from_folder(test_path)
    validate_images = load_images_from_folder(val_path)

    return train_images, test_images, validate_images


# Function to retrieve the image slices and provide a summary with a table and example images
def get_image_slices_with_table():
    train_images, test_images, validate_images = get_image_slices()

    # Display a summary table using PrettyTable
    table = PrettyTable()
    table.field_names = ["Data Split", "Total Images", "Image Shape"]
    table.add_row(["Training", len(train_images), train_images[0].shape])
    table.add_row(["Testing", len(test_images), test_images[0].shape])
    table.add_row(["Validation", len(validate_images), validate_images[0].shape])

    print(table)

    # Plot an example image from each dataset split
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(train_images[0], cmap='gray')
    axs[0].set_title("Training Image")
    axs[0].axis('off')

    axs[1].imshow(test_images[0], cmap='gray')
    axs[1].set_title("Testing Image")
    axs[1].axis('off')

    axs[2].imshow(validate_images[0], cmap='gray')
    axs[2].set_title("Validation Image")
    axs[2].axis('off')

    plt.show()

    return train_images, test_images, validate_images

# Call the function to display the dataset summary and example images
get_image_slices_with_table()