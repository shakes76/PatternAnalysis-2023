import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self, root_dir):
        super(SiameseNetworkDataset, self).__init__()

        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomRotation(degrees=(0, 60)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.2)),
            transforms.RandomPerspective(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

        # Define categories
        self.categories = ['AD', 'NC']

        # Load images for each category
        self.images = {category: [] for category in self.categories}
        for category in self.categories:
            for img_filename in os.listdir(os.path.join(root_dir, category)):
                with Image.open(os.path.join(root_dir, category, img_filename)) as img:
                    self.images[category].append(self.transform(img.copy()))

        # Stack images into tensors for each category
        for category in self.categories:
            self.images[category] = torch.stack(self.images[category])

    def __len__(self):
        return min(len(self.images['AD']), len(self.images['NC']))

    def __getitem__(self, index):
        if random.choice([True, False]):
            # Positive example (both images are AD)
            img1 = random.choice(self.images['AD'])
            img2 = random.choice(self.images['AD'])
            label = torch.tensor(1, dtype=torch.float)
        else:
            # Negative example (one image is AD, the other is NC)
            img1 = random.choice(self.images['AD'])
            img2 = random.choice(self.images['NC'])
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label


def get_datasets(root_dir):
    train_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'train'))
    test_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'test'))
    val_dataset = SiameseNetworkDataset(os.path.join(root_dir, 'val'))
    return train_dataset, test_dataset, val_dataset
