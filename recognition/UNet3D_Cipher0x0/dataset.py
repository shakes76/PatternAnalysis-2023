import glob
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import nibabel as nib


class Prostate_3D(Dataset):
    def __init__(self, data_img_dir, label_img_dir, transform=None, target_transform=None):
        self.data_img_path = glob.glob(data_img_dir + '*.nii.gz')
        self.label_img_path = glob.glob(label_img_dir + '*.nii.gz')
        #self.data = nib.load(self.data_img_path).get_fdata()
        #self.label = nib.load(self.label_img_path).get_fdata()
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.label_img_path)

    def __getitem__(self, idx):
        data = torch.from_numpy(nib.load(self.data_img_path[idx]).get_fdata()).unsqueeze(0)
        label = torch.from_numpy(nib.load(self.label_img_path[idx]).get_fdata()).unsqueeze(0)
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        return data, label