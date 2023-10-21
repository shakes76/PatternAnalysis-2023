import os
import random
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader

TRAIN_FILE_ROOT = "./AD_NC/train"
TRAIN_AD = "./AD_NC/train/AD"
TRAIN_NC = "./AD_NC/train/NC"
BATCH_SIZE = 64
VAL_SIZE = 0.1
TRAIN_SIZE = 0.9


class SiameseDataSet(Dataset):
    """
    Class for loading the ADNI dataset and retrieving the triplet image input
    """

    def __init__(self, imgset: datasets.ImageFolder, transform=None):
        self.imgset = imgset
        self.transform = transform

    def __getitem__(self, item):
        # indexing
        pass

    def __len__(self):
        # return n_samples
        pass


def get_patients(path: str) -> [str]:
    uids = []
    dire = os.fsdecode(path)
    for file in os.listdir(dire):
        filename = os.fsdecode(file)
        substrings = filename.split("_", 2)
        if substrings[0] not in uids:
            uids.append(substrings[0])

    # print(f"{path} : {len(uids)}")
    return uids


def remove_patients(imgset: datasets.ImageFolder, index: int, match_set: []) -> datasets.ImageFolder:
    folder = ""
    if index == 0:
        folder = "AD"
    else:
        folder = "NC"

    for patient in match_set:
        for fname in os.listdir(TRAIN_FILE_ROOT + "/" + folder):
            if patient in fname:
                imgset.imgs.remove((TRAIN_FILE_ROOT + "\\" + folder + "\\" + fname, index))

    return imgset


def patient_split() -> (datasets.ImageFolder, datasets.ImageFolder):
    files = get_patients(TRAIN_AD)
    random.shuffle(files)
    train_ad, validate_ad = np.split(files, [int(len(files) * TRAIN_SIZE)])
    files = get_patients(TRAIN_NC)
    random.shuffle(files)
    train_nc, validate_nc = np.split(files, [int(len(files) * TRAIN_SIZE)])

    train_dataset = datasets.ImageFolder(root=TRAIN_FILE_ROOT)
    validation_dataset = datasets.ImageFolder(root=TRAIN_FILE_ROOT)

    train_dataset = remove_patients(train_dataset, 0, validate_ad)
    train_dataset = remove_patients(train_dataset, 1, validate_nc)

    validation_dataset = remove_patients(validation_dataset, 0, train_ad)
    validation_dataset = remove_patients(validation_dataset, 1, train_nc)

    return train_dataset, validation_dataset


if __name__ == "__main__":
    patient_split()
