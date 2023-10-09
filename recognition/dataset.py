import os
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2

class ISICDataset(Dataset):
    def __init__(self, path, type, desired_height=256, desired_width=256):

        """

        :param path: path to the dataset directory
        :param type: type of dataset, either "Training" or "Validation" or "Testing"
        :param desired_height: height of the image after resizing
        :param desired_width: width of the image after resizing
        """

        self.path = path
        self.type = type
        self.image_folder = f"{path}/{type}_Input"
        self.mask_folder = f"{path}/{type}_GroundTruth"
        self.transform = transforms.Compose([
            transforms.Resize((desired_height, desired_width)),
            transforms.ToTensor()
        ])

        # Check if the dataset is valid
        try:
            self.image_filenames = [f for f in os.listdir(self.image_folder) if f.endswith('.jpg')]
            self.mask_filenames = [f for f in os.listdir(self.mask_folder) if f.endswith('.png')]
        except FileNotFoundError:
            raise FileNotFoundError(f"The folder path '{path}' does not exist.")

        # Check if the images and masks are in a one-to-one correspondence
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        image = Image.fromarray(image)
        mask = Image.fromarray(mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# if __name__ == '__main__':
#     path = "E:/comp3710/ISIC2018"
#     train_dataset = ISICDataset(path, "Training")
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     for images, masks in train_loader:
#         print(images.shape, masks.shape)
