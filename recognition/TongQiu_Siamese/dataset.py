import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import random
import os

from utils import Config


def discover_directory(root_dir):
    classes = ['AD', 'NC']
    data = []
    for cls in classes:
        class_path = os.path.join(root_dir, cls)
        for img_name in sorted(os.listdir(class_path)):
            img_path = os.path.join(class_path, img_name)
            patient_id = img_name.split('_')[0]
            data.append((img_path, cls, patient_id))
    return data


def patient_level_split(full_data, ratio_train=0.8):
    unique_patients = list(set(patient_id for _, _, patient_id in full_data))
    random.shuffle(unique_patients)
    split_idx = int(ratio_train * len(unique_patients))

    train_patients = unique_patients[:split_idx]
    val_patients = unique_patients[split_idx:]

    train_data = [(img, label, patient_id) for img, label, patient_id in full_data if patient_id in train_patients]
    val_data = [(img, label, patient_id) for img, label, patient_id in full_data if patient_id in val_patients]

    return train_data, val_data


class ContrastiveDataset(Dataset):
    def __init__(self, data_lst, transform=None):
        self.data_lst = data_lst
        self.transform = transform
        self.classes = ['AD', 'NC']

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        img_path, cls, patient_id = self.data_lst[idx]
        same_class = random.random() > 0.5
        if same_class:
            other_patients = [(i, c, p) for i, c, p in self.data_lst if c == cls and p != patient_id]
            other_img_path, _, other_patient_id = random.choice(other_patients)
        else:
            other_cls = 'NC' if cls == 'AD' else 'AD'
            other_patients = [(i, c, p) for i, c, p in self.data_lst if c == other_cls and p != patient_id]
            other_img_path, _, other_patient_id = random.choice(other_patients)

        # Read images
        img1 = read_image(img_path, ImageReadMode.GRAY).float()
        img2 = read_image(other_img_path, ImageReadMode.GRAY).float()
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = torch.tensor([1 if same_class else 0], dtype=torch.float32)

        return img1, img2, label


class TripletDataset(Dataset):
    pass


"""
from torch.utils.data import DataLoader
if __name__ == '__main__':

    # patient-level split
    full_train_data = discover_directory(Config.TRAIN_DIR)
    train_data, val_data = patient_level_split(full_train_data)

    train_dataset = ContrastiveDataset(train_data)
    val_dataset = ContrastiveDataset(val_data)


    dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=3,
        num_workers=1,
        drop_last=True
    )
    for batch in dataloader:
        print("Batch:")
        print("volume1 shape:", batch[0].shape)
        print("volume2 shape:", batch[1].shape)
        print("label:", batch[2].shape)
        print()
        break
"""
