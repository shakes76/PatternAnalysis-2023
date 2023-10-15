import torch
import os
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
from torchvision import transforms

AD_PATH = 'R:\\pattern\\PatternAnalysis-2023\\recognition\\s48238702\\AD_NC\\train\\AD'
CN_PATH = 'R:\\pattern\\PatternAnalysis-2023\\recognition\\s48238702\\AD_NC\\train\\NC'
AD_TEST_PATH = 'R:\\pattern\\PatternAnalysis-2023\\recognition\\s48238702\\AD_NC\\test\\AD'
CN_TEST_PATH = 'R:\\pattern\\PatternAnalysis-2023\\recognition\\s48238702\\AD_NC\\test\\NC'

class SiameseDataset(Dataset):
    def __init__(self, pairs, labels, transform=None):
        self.pairs = pairs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        label = self.labels[idx]

        image1 = Image.open(pair[0])
        image2 = Image.open(pair[1])

        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, label

    def __len__(self):
        return len(self.pairs)

def load_siamese_data(batch_size=32):
    ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
    cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]


def ImageToTensor(image):
    image = image.convert('L')
    image = image.resize((128, 128))
    image = np.array(image)
    image = image.reshape((1, 128, 128))
    image = torch.from_numpy(image).float()
    image = image.div(255)
    return image
