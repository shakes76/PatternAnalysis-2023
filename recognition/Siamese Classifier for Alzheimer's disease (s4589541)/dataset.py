"""
    dataset.py - loading and preprocessing of data
"""
import random
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import show_image, train_valid_split

BATCH_SIZE = 32

class OneChannel:
    """Custom transform class to discard extra image channels."""
    @staticmethod
    def __call__(img_tensor):
        return torch.unsqueeze(img_tensor[0], dim=0)


class TripletDataset(datasets.ImageFolder):
    """Custom dataset for providing image triplets to Siamese model."""
    def __init__(self, root, transform=None, triplet=True):
        super().__init__(root, transform=transform)
        self.triplet = triplet

    def __getitem__(self, index: int):
        """Overrides inherited method. Returns sample image (anchor) and class, with positive and negative image.

        Args:
            index (int): Index

        Returns:
            tuple: (anchor, target, positive, negative) where target is class_index of the target class.
        """
        # first, get the anchor image 
        path, target = self.samples[index]
        anchor = self.loader(path)
        # print(f"Anchor: {path}")
        if not self.triplet:
            anchor = self.transform(anchor)
            return anchor, target

        # now get the positive and negative images randomly
        found_p = False
        found_n = False
        num_samples = len(self.samples)
        while not (found_p and found_n):
            rand_idx = random.randint(0, num_samples-1)
            if rand_idx == index:
                continue
            new_path, new_target = self.samples[rand_idx]
            if not found_p and new_target == target:
                positive = self.loader(new_path)
                found_p = True
                # print(f"Positive: {new_path}")
            elif not found_n and new_target != target:
                negative = self.loader(new_path)
                found_n = True
                # print(f"Negative: {new_path}")

        if self.transform is not None:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, target, positive, negative


if __name__ == '__main__':
    # split the training data into training and validation datasets
    # create new data directory
    train_valid_split()
