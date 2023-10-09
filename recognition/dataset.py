import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset


class ISICDataset(Dataset):
    def __init__(self, path, type, transform=None):

        """

        :param path: path to folder containing images and masks
        :param type: type of dataset (Training, Validation, Test)
        :param transform: transform to apply to images and masks
        """

        self.path = path
        self.type = type
        self.transform = transform
        self.image_folder = f"{path}/{type}_Input"
        self.mask_folder = f"{path}/{type}_GroundTruth"

        try:
            self.image_filenames = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
            self.mask_filenames = [f for f in os.listdir(self.mask_folder) if f.endswith('.png')]
        except FileNotFoundError:
            raise FileNotFoundError(f"The folder path '{path}' does not exist.")

        # Ensure each image corresponds to a mask
        image_basenames = set([os.path.splitext(f)[0] for f in self.image_filenames])
        mask_basenames = set([os.path.splitext(f)[0].replace('_segmentation', '') for f in self.mask_filenames])
        if image_basenames != mask_basenames:
            raise ValueError("The images and masks are not in a one-to-one correspondence.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_filenames[idx])
        mask_path = os.path.join(self.mask_folder, self.mask_filenames[idx])

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # Assuming the mask is a grayscale image

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


if __name__ == '__main__':
    path = "E:/comp3710/ISIC2018"
    train_dataset = ISICDataset(path, "Training")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    for images, masks in train_loader:
        print(images.shape, masks.shape)
