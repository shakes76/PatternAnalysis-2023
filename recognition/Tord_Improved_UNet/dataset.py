import torchvision.transforms as transforms
import torchvision.datasets.vision as VisionDataset
from torch.utils.data import DataLoader
import torchvision
import torch
import os
from torchvision.io import read_image
from torchvision.datasets import ImageFolder

#Dataloader and preprocessing for the dataset

class TrainingDataset(VisionDataset):
    def __init__(self, root, data_transform=None, target_transform=None):
        super(TrainingDataset, self).__init__(root, data_transform=data_transform, target_transform=target_transform)
        self.data_transform = data_transform
        self.target_transform = target_transform
        
        self.images = []
        self.masks = []

        data_folder = os.path.join(root, 'ISIC2018_Task1-2_Training_Input')
        mask_folder = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth')

        for image_name in os.listdir(data_folder):
            mask_name = image_name.replace('.jpg', '_segmentation.png')  # Assumes naming conventions for masks
            self.images.append(os.path.join(data_folder, image_name))
            self.masks.append(os.path.join(mask_folder, mask_name))
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        image = read_image(image_path)
        mask = read_image(mask_path)

        if self.data_transform is not None:
            image = self.data_transform(image)

        if self.target_transform is not None:
            mask = self.target_transform(mask)

        return image, mask

data_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
target_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
trainingset = TrainingDataset(root='./data', data_transform=data_transform, target_transform=target_transform)
print(trainingset.__len__())