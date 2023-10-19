import torch
import os
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
from PIL import Image
from torchvision import transforms

GLOBAL_PATH = 'R:\pattern\PatternAnalysis-2023\recognition\s48238702'
AD_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'train', 'AD')
CN_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'train', 'NC')
AD_TEST_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'test', 'AD')
CN_TEST_PATH = os.path.join(GLOBAL_PATH, 'AD_NC', 'test', 'NC')

class SiameseDataset(Dataset):
    def __init__(self, path1, path2, labels, transform=None):
        self.path1 = path1
        self.path2 = path2
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.path1)

    def __getitem__(self, idx):
        img1 = Image.open(self.path1[idx]).convert('L')
        img2 = Image.open(self.path2[idx]).convert('L')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        label = self.labels[idx]

        return img1, img2, label
def load_siamese_data(batch_size=32):
    pair_base = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)][::2]
    pair_ad = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)][:len(pair_base)]
    pair_cn = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)][1::4][:len(pair_base)]
    pair_compare = pair_cn + pair_ad
    labels = np.concatenate([np.zeros(len(pair_base)//2), np.ones(len(pair_base)//2)])

    transform = transforms.Compose([transforms.Resize((128, 128)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    siamese_dataset = SiameseDataset(pair_base, pair_compare, labels, transform=transform)
    dataloader = DataLoader(siamese_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

def load_classify_data(testing: bool, batch_size=32):
    """ Load testing image data, images with labels,
    0 for ad, 1 for cn

    Args:
        testing (bool): data for testing
        batch_size (int): batch size for the data

    Returns:
        (train_loader, val_loader): tuple of DataLoader for training and validation data
    """
    # Get the path to each image and mask
    if (not testing):
        ad_paths = [os.path.join(AD_PATH, path) for path in os.listdir(AD_PATH)]
        cn_paths = [os.path.join(CN_PATH, path) for path in os.listdir(CN_PATH)]
    else:
        ad_paths = [os.path.join(AD_TEST_PATH, path) for path in os.listdir(AD_TEST_PATH)]
        cn_paths = [os.path.join(CN_TEST_PATH, path) for path in os.listdir(CN_TEST_PATH)]

    paths = ad_paths + cn_paths

    # Create dataset with 2 classes, ad and cn
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
