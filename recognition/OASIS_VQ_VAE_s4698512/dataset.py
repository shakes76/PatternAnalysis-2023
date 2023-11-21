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
from torch.utils.data import DataLoader


TRAIN_BATCH_SIZE = 76
TEST_BATCH_SIZE = 76
VALIDATE_BATCH_SIZE = 76

# OASIS Data set
TYPE = "seg_"       # "seg_" for segmented, or "" for non-segmented
DATASETS_PATH = os.path.join("..", "PatternAnalysisData", "datasets")
OASIS_TRANS_DIR = os.path.join(DATASETS_PATH, "OASIS_processed")

FOLDS = ["train", "test", "validate"]
OASIS_FOLD_PATH = {(TYPE, FOLD): f"keras_png_slices_{TYPE}{FOLD}" for TYPE in [
    "seg_", ""] for FOLD in FOLDS}


# ADNI Dataset
ADNI_TRANS_DIR = os.path.join(DATASETS_PATH, "ADNI_processed")
CASES = ["AD", "NC"]    # Alzheimer's or control
# Adni dataset doesn't contain validation
ADNI_FOLD_PATH = {
    (FOLD, CASE): os.path.join(FOLD, CASE) for FOLD in FOLDS[0:2] for CASE in CASES}


class OASIS:
    """
    Defines data importer and data transforms as well as the data loader for the OASIS dataset.
    """

    def __init__(self, root_dir, transform=None, copy=False):
        """
        Initializes the OASIS class.

        Args:
            root_dir (str): The root directory containing the OASIS dataset.
            transform (callable, optional): A function/transform to apply to the data (default: None).
            copy (bool, optional): If True, the data will be copied to a processed folder and separated by patient (default: False).
        """

        self.root_dir = root_dir
        self.transfrom = transform

        # Copies imagess into their own case folder if not already done
        if copy:
            for fold in FOLDS:
                source_dir = os.path.join(
                    self.root_dir, OASIS_FOLD_PATH[(TYPE, fold)])

                for filename in os.listdir(source_dir):
                    if not filename.endswith(".png"):
                        continue    # skip file

                    # Assumes the format 'seg_xxx_slice_y.nii.png'
                    patient_id = filename.split("_")[1]

                    # Define the destination directory for this patient
                    destination_dir = os.path.join(
                        OASIS_TRANS_DIR, OASIS_FOLD_PATH[(TYPE, fold)], patient_id)

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


class ADNI:
    """
    Defines data importer and data transforms as well as the data loader for the ADNI dataset.
    """

    def __init__(self, root_dir: str, sampleType: str, transform=None, copy: bool = False) -> None:
        """
        Initializes the ADNI class.

        Args:
            root_dir (str): The root directory containing the ADNI dataset.
            sampleType (str): AD if Alzheimer's samples to be generated; NC if control samples to be generated
            transform (callable, optional): A function/transform to apply to the data (default: None).
            copy (bool, optional): If True, the data will be copied to a processed folder and separated by patient (default: False).
        """

        self.root_dir = root_dir
        self.transfrom = transform

        # Copies imagess into their own case folder if not already done
        if copy:
            # Loop over train and test sets
            for fold in FOLDS[0:2]:
                # Define source directory for this fold
                source_dir = os.path.join(
                    self.root_dir, ADNI_FOLD_PATH[(fold, sampleType)])

                # Loop over files in source directory.
                for filename in os.listdir(source_dir):

                    if not filename.endswith(".jpeg"):
                        continue    # skip file if it's not a jpeg

                    # Extract patient ID from filename
                    # Assumes the format 'xxxxx_yy.jpeg' where xxxxx is pt id and yy is slice
                    patient_id = filename.split("_")[0]

                    # Define the destination directory for this patient
                    destination_dir = os.path.join(
                        ADNI_TRANS_DIR, ADNI_FOLD_PATH[(fold, sampleType)], patient_id)

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
            ADNI_TRANS_DIR, ADNI_FOLD_PATH[("train", sampleType)])
        TEST_DIR = os.path.join(
            ADNI_TRANS_DIR, ADNI_FOLD_PATH[("test", sampleType)])

        # Load Dataset
        self.train_dataset = ImageFolder(TRAIN_DIR, transform=transform)
        self.test_dataset = ImageFolder(TEST_DIR, transform=transform)

        self.train_loader = DataLoader(
            self.train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(
            self.test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)

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


# Test script
if __name__ == "__main__":
    dataset = "ADNI"

    if dataset == "OASIS":
        oasis_data_path = os.path.join(DATASETS_PATH, "OASIS")
        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Adjust the size as needed
            transforms.ToTensor(),
        ])

        # Check if "OASIS_processed" folder does not exist
        oasis_processed_folder = os.path.join(DATASETS_PATH, "OASIS_processed")
        copy = not os.path.exists(oasis_processed_folder)

        oasis = OASIS(oasis_data_path, transform=transform, copy=copy)

    elif dataset == "ADNI":
        adni_data_path = os.path.join(DATASETS_PATH, "ADNI", "AD_NC")
        # Define a transformation to apply to the images (e.g., resizing)
        transform = transforms.Compose([
            transforms.Resize((240, 256)),  # Adjust the size as needed
            transforms.ToTensor(),
        ])

        # Check if "ADNI_processed" folder does not exist
        adni_processed_folder = os.path.join(DATASETS_PATH, "ADNI_processed")
        copy = not os.path.exists(adni_processed_folder)

        adni = ADNI(adni_data_path, transform=transform, copy=copy)
