import os
import random
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir):
        super(SiameseNetworkDataset, self).__init__()

        self.transform = transforms.ToTensor()

        # Load AD and NC images
        self.AD_images_path = os.path.join(root_dir, 'AD')
        self.NC_images_path = os.path.join(root_dir, 'NC')

        # Get list of image filenames
        self.AD_image_filenames = os.listdir(self.AD_images_path)
        self.NC_image_filenames = os.listdir(self.NC_images_path)

        # Load images and transform them to tensors
        self.AD_images = [self.transform(Image.open(os.path.join(self.AD_images_path, img_filename)).convert("L")) for img_filename in self.AD_image_filenames]
        self.NC_images = [self.transform(Image.open(os.path.join(self.NC_images_path, img_filename)).convert("L")) for img_filename in self.NC_image_filenames]

        # Stack images into tensors
        self.AD_images_tensor = torch.stack(self.AD_images)
        self.NC_images_tensor = torch.stack(self.NC_images)

    def __len__(self):
        return 2 * min(len(self.AD_images_tensor), len(self.NC_images_tensor))  # 2 times for positive and negative pairs

    def __getitem__(self, index):
        if random.choice([True, False]):
            # Positive example (both images are AD)
            img1 = Image.open(os.path.join(self.AD_images_path, random.choice(self.AD_image_filenames))).convert("L")
            img2 = Image.open(os.path.join(self.AD_images_path, random.choice(self.AD_image_filenames))).convert("L")
            label = torch.tensor(1, dtype=torch.float)
        else:
            # Negative example (one image is AD, the other is NC)
            img1 = Image.open(os.path.join(self.AD_images_path, random.choice(self.AD_image_filenames))).convert("L")
            img2 = Image.open(os.path.join(self.NC_images_path, random.choice(self.NC_image_filenames))).convert("L")
            label = torch.tensor(0, dtype=torch.float)

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label


def transform_process():
    # Define the transformations to be applied on the images.
    current_transform = transforms.Compose([
        # transforms.Resize((105, 105)),
        # transforms.CenterCrop(105),
        transforms.RandomRotation(degrees=(0, 60)),  # Randomly rotate the image by 30 degrees
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Randomly change the brightness and contrast of an image
        transforms.ToTensor(),  # Convert the image to PyTorch tensor data type
        transforms.Normalize(mean=[0.485], std=[0.229]),
        # transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),  # Randomly resize the cropped image
        # transforms.Lambda(convert_to_rgb),  # Convert the image to RGB if it is a grayscale image.
    ])
    return current_transform


def transform_directory():
    transform = transform_process()
    # Use relative paths to specify the directories.
    current_train_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/train"))
    current_test_dir = ImageFolder(root=os.path.join(os.getcwd(), "AD_NC/test"))

    # Use relative paths to specify the directories.
    current_train_dataset = SiameseNetworkDataset(root_dir=os.path.join(os.getcwd(), "AD_NC/train"))
    current_test_dataset = SiameseNetworkDataset(root_dir=os.path.join(os.getcwd(), "AD_NC/test"))

    return current_train_dir, current_test_dir, current_train_dataset, current_test_dataset
