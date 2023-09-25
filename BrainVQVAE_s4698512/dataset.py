"""
Hugo Burton
s4698512
20/09/2023

dataset.py
Contains the data loader for the ADNI dataset as well as performing preprocessing
"""

import os
import shutil
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

TYPE = "seg_"       # "seg_" for segmented, or "" for non-segmented

COPIED = True
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 128
VALIDATE_BATCH_SIZE = 128

OASIS_TRANS_DIR = os.path.join(".", "datasets", "OASIS_processed")

FOLDS = ["train", "test", "validate"]
OASIS_FOLD_PATH = {(TYPE, FOLD): f"keras_png_slices_{TYPE}{FOLD}" for TYPE in [
    "seg_", ""] for FOLD in FOLDS}


class OASIS():
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transfrom = transform

        # Copies imagess into their own case folder if not already done
        if not COPIED:
            for fold in FOLDS:
                source_dir = os.path.join(
                    self.root_dir, OASIS_FOLD_PATH[(TYPE, fold)])

                for filename in os.listdir(source_dir):
                    if not filename.endswith(".png"):
                        continue    # skip file
                    # print(filename)
                    # Assumes the format 'seg_xxx_slice_y.nii.png'
                    patient_id = filename.split("_")[1]
                    # print(patient_id)

                    # Define the destination directory for this patient
                    destination_dir = os.path.join(
                        OASIS_TRANS_DIR, OASIS_FOLD_PATH[(TYPE, fold)], patient_id)
                    # print(destination_dir)

                    # Create the destination directory if it doesn't exist
                    os.makedirs(destination_dir, exist_ok=True)
                    print("made directory", destination_dir)
                    # Define the source and destination file paths
                    source_filepath = os.path.join(source_dir, filename)
                    destination_filepath = os.path.join(
                        destination_dir, filename)

                    # Copy the image from source to destination
                    shutil.copy(source_filepath, destination_filepath)
                    print("copying", source_dir, "to", destination_dir)

        # New file paths
        TRAIN_DIR = os.path.join(
            OASIS_TRANS_DIR, OASIS_FOLD_PATH[(TYPE, "train")])
        TEST_DIR = os.path.join(
            OASIS_TRANS_DIR, OASIS_FOLD_PATH[(TYPE, "test")])
        VALIDATE_DIR = os.path.join(
            OASIS_TRANS_DIR, OASIS_FOLD_PATH[(TYPE, "validate")])

        # Load Dataset
        self.train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
        self.test_dataset = ImageFolder(TEST_DIR, transform=transform)
        self.validate_dataset = ImageFolder(VALIDATE_DIR, transform=transform)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
        self.validate_loader = DataLoader(
            self.validate_dataset, batch_size=VALIDATE_BATCH_SIZE, shuffle=False)

        # print("Testing train loader")
        # Iterate through the DataLoader to get a batch of data
        # for batch_idx, data in enumerate(self.train_loader):
        #     # "data" will contain a tuple (or dictionary) of inputs and labels, if applicable
        #     inputs, labels = data  # Modify this if your DataLoader provides different data structure

        #     # Print information about the batch
        #     print(f"Batch {batch_idx + 1}:")
        #     # Number of samples in the batch
        #     print(f"Batch size: {len(inputs)}")

        #     # Print or process the data as needed
        #     # Shape of the input data (e.g., [batch_size, channels, height, width])
        #     print("Input data shape:", inputs.shape)

        #     # Optionally, print or process labels if your dataset has labels
        #     if labels is not None:
        #         # Shape of the labels (e.g., [batch_size])
        #         print("Labels shape:", labels.shape)

        #     # Optionally, visualize or inspect a few samples from the batch
        #     # For example, if you are working with image data:
        #     import matplotlib.pyplot as plt

        #     # Plot the first few images from the batch
        #     num_samples_to_plot = 4
        #     for i in range(min(num_samples_to_plot, len(inputs))):
        #         plt.subplot(1, num_samples_to_plot, i + 1)
        #         # Assuming channels-last format for images
        #         plt.imshow(inputs[i].permute(1, 2, 0).numpy())
        #         plt.title(f"Sample {i + 1}")
        #         plt.axis("off")

        #     plt.show()

        #     # Break the loop if you only want to inspect the first batch
        #     break

    def __len__(self):
        """
        Returns the number of images in the train, test and validate sets as tuple
        """

        return len(self.train_dataset), len(self.test_dataset), len(self.validate_dataset)

    def __get__item(self, index):
        """
        Returns a specific image given the index
        """
        # Transform
        if self.transform:
            image = self.transfrom(image)


class ADNI():
    def __init__(self, root_dir, transform=None) -> None:
        self.root_dir = root_dir
        self.transfrom = transform

        self.train_dir_ad = "train/AD/"
        self.train_dir_nc = "train/NC/"

    def __len__(self):
        """
        Returns the number of images in the dataset
        """
        pass

    def __get__item(self, index):
        """
        Returns a specific image / target to image
        """
        # Transform
        if self.transform:
            image = self.transfrom(image)


# Test
if __name__ == "__main__":
    oasis_data_path = os.path.join(".", "datasets", "OASIS")
    # Define a transformation to apply to the images (e.g., resizing)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust the size as needed
        transforms.ToTensor(),
    ])
    oasis = OASIS(oasis_data_path, transform=transform)
