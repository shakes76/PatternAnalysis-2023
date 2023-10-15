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


def load_classify_data(testing: bool, batch_size=32):
    
    if (not testing):
        ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
        cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]
    else:
        ad_paths = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
        cn_paths = [os.path.join(CN_TEST_PATH, path) for path in os.listdir(CN_TEST_PATH)]

    paths = ad_paths + cn_paths

    labels = [0 if path.endswith('AD') else 1 for path in paths]

    dataset = []
    for i in range(len(paths)):
        image = Image.open(paths[i])
        image_tensor = ImageToTensor(image)

        label = labels[i]

        dataset.append((image_tensor, label))

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def ImageToTensor(image):
    image = image.convert('L')
    image = image.resize((128, 128))
    image = np.array(image)
    image = image.reshape((1, 128, 128))
    image = torch.from_numpy(image).float()
    image = image.div(255)
    return image
