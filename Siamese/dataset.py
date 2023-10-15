import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ADNIDataset(Dataset):
    def __init__(self, data_path):
        super(ADNIDataset, self).__init__()

        self.transform = transforms.ToTensor()

        self.ad_path = os.path.join(data_path, 'AD')
        self.nc_path = os.path.join(data_path, 'NC')

        self.ad_images_list = [Image.open(os.path.join(self.ad_path, img)) for img in os.listdir(self.ad_path)]
        self.nc_images_list = [Image.open(os.path.join(self.nc_path, img)) for img in os.listdir(self.nc_path)]

        self.ad_images = torch.stack([transforms.ToTensor()(img) for img in self.ad_images_list])
        self.nc_images = torch.stack([transforms.ToTensor()(img) for img in self.nc_images_list])

    def __len__(self):
        return min(len(self.ad_images), len(self.nc_images))

    def __getitem__(self, index):
        if index % 2 == 0:
            # Positive example (both images are AD)
            img1 = self.ad_images[index % len(self.ad_images)]
            img2 = self.ad_images[(index + 1) % len(self.ad_images)]
            label = torch.tensor(1, dtype=torch.float)
        else:
            # Negative example (one image is AD, the other is NC)
            img1 = self.ad_images[index % len(self.ad_images)]
            img2 = self.nc_images[index % len(self.nc_images)]
            label = torch.tensor(0, dtype=torch.float)

        return img1, img2, label


def get_patient_ids(dataset):
    patient_ids = set()
    for img_path in dataset:  # 这里你需要遍历你的图像文件路径
        patient_id = img_path.split('_')[0]  # 提取病人 ID，假设它在文件名的第一个字段
        patient_ids.add(patient_id)
    return list(patient_ids)


def split_patient_ids(patient_ids, val_ratio=0.2):
    num_val = int(len(patient_ids) * val_ratio)
    return patient_ids[num_val:], patient_ids[:num_val]


def get_indices_from_patient_ids(patient_ids, dataset):
    indices = []
    for index in range(len(dataset)):
        img1, img2, _ = dataset[index]
        # 提取病人 ID 的逻辑。这里假设 img1 和 img2 的文件名中包含病人 ID。
        # 你需要根据你的具体情况进行修改。
        patient_id1 = img1.split('_')[0]
        patient_id2 = img2.split('_')[0]

        if patient_id1 in patient_ids or patient_id2 in patient_ids:
            indices.append(index)
    return indices


def get_train_dataset(data_path):
    train_dataset = ADNIDataset(os.path.join(data_path, 'train'))
    return train_dataset


def get_test_dataset(data_path):
    test_dataset = ADNIDataset(os.path.join(data_path, 'test'))
    return test_dataset
