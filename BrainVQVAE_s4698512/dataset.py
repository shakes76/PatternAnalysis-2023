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

        # TRAIN_PATH = [os.path.join(self.root_dir, "keras_png_slices_seg_train"),
        #               os.path.join(self.root_dir, "keras_png_slices_train")][TYPE]
        # TEST_PATH = [os.path.join(self.root_dir, "keras_png_slices_seg_test"),
        #              os.path.join(self.root_dir, "keras_png_slices_test")][TYPE]
        # VALID_PATH = [os.path.join(self.root_dir, "keras_png_slices_seg_validate"),
        #               os.path.join(self.root_dir, "keras_png_slices_validate")][TYPE]

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
                destination_filepath = os.path.join(destination_dir, filename)

                # Copy the image from source to destination
                shutil.copy(source_filepath, destination_filepath)
                print("copying", source_dir, "to", destination_dir)

        # self.train_image_paths = [os.path.join(
        #     TRAIN_PATH, filename) for filename in os.listdir(TRAIN_PATH)]

        # self.valid_image_paths = [os.path.join(
        #     VALID_PATH, filename) for filename in os.listdir(VALID_PATH)]

        # self.test_image_paths = [os.path.join(
        #     TEST_PATH, filename) for filename in os.listdir(TEST_PATH)]

        # # Load Dataset
        # train_dataset = ImageFolder(os.path.join(
        #     self.root_dir, TEST_PATH), transform=transform)
        # test_dataset = ImageFolder(os.path.join(
        #     self.root_dir, TRAIN_PATH), transform=transform)
        # validate_dataset = ImageFolder(os.path.join(
        #     self.root_dir, VALID_PATH), transform=transform)

        # train_loader = DataLoader(
        #     train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
        # test_loader = DataLoader(
        #     test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
        # validate_loader = DataLoader(
        #     validate_dataset, batch_size=VALIDATE_BATCH_SIZE, shuffle=False)

        print("Here, loaded")

    def __len__(self):
        """
        Returns the number of images in the dataset
        """
        pass

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
