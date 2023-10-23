import torch
from torch.utils.data import Dataset
from torchvision.io import read_image, ImageReadMode
import random
import os
import torchvision.transforms as tf

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
        img1 = read_image(img_path, ImageReadMode.GRAY).float()/255.
        img2 = read_image(other_img_path, ImageReadMode.GRAY).float()/255.
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        label = torch.tensor([1 if same_class else 0], dtype=torch.float32)

        return img1, img2, label


class TripletDataset(Dataset):
    def __init__(self, data_lst, transform=None):
        self.data_lst = data_lst
        self.transform = transform
        self.classes = ['AD', 'NC']

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        anchor_img_path, anchor_cls, anchor_patient_id = self.data_lst[idx]

        # Positive selection
        positive_samples = [(i, c, p) for i, c, p in self.data_lst if c == anchor_cls and p != anchor_patient_id]
        positive_img_path, _, positive_patient_id = random.choice(positive_samples)

        # Negative selection
        negative_cls = 'NC' if anchor_cls == 'AD' else 'AD'
        negative_samples = [(i, c, p) for i, c, p in self.data_lst if c == negative_cls and p != anchor_patient_id]
        negative_img_path, _, negative_patient_id = random.choice(negative_samples)

        # Read images
        anchor_img = read_image(anchor_img_path, ImageReadMode.GRAY).float() / 255.
        positive_img = read_image(positive_img_path, ImageReadMode.GRAY).float() / 255.
        negative_img = read_image(negative_img_path, ImageReadMode.GRAY).float() / 255.
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        label = torch.tensor([1 if anchor_cls == 'AD' else 0], dtype=torch.float32)

        return anchor_img, positive_img, negative_img, label


class ClassificationDataset(Dataset):
    def __init__(self, data_lst, transform=None):
        self.data_lst = data_lst
        self.transform = transform
        self.class_to_tensor = {'AD': torch.tensor([0], dtype=torch.float32),
                                'NC': torch.tensor([1], dtype=torch.float32)}

    def __len__(self):
        return len(self.data_lst)

    def __getitem__(self, idx):
        anchor_img_path, anchor_cls, anchor_patient_id = self.data_lst[idx]
        anchor_cls_tensor = self.class_to_tensor[anchor_cls]

        # Read images
        anchor_img = read_image(anchor_img_path, ImageReadMode.GRAY).float() / 255.
        if self.transform:
            anchor_img = self.transform(anchor_img)

        return anchor_img, anchor_cls_tensor


from torch.utils.data import DataLoader
if __name__ == '__main__':

    # patient-level split
    full_train_data = discover_directory(Config.TRAIN_DIR)
    train_data, val_data = patient_level_split(full_train_data)

    tr_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])

    val_transform = tf.Compose([
        tf.Normalize((0.1160,), (0.2261,)),
        tf.RandomRotation(10)
    ])

    train_dataset = ClassificationDataset(train_data, transform=tr_transform)
    val_dataset = ClassificationDataset(val_data, transform=val_transform)

    dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=3,
        num_workers=1,
        drop_last=True
    )
    for batch in dataloader:
        print("Batch:")
        print("anchor shape:", batch[0].shape)
        print("positive shape:", batch[1].shape)
        print("positive shape:", batch[1].dtype)
        break

