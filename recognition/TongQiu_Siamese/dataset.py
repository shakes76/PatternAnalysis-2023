import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import random
import os

from utils import Config


class ContrastiveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['AD', 'NC']

        self.patient_volumes = {'AD': {}, 'NC': {}}
        for cls in self.classes:
            class_path = os.path.join(root_dir, cls)
            for img_name in sorted(os.listdir(class_path)):  # Sorting ensures slices are in order
                patient_id = img_name.split('_')[0]
                if patient_id not in self.patient_volumes[cls]:
                    self.patient_volumes[cls][patient_id] = []
                slice_img = os.path.join(class_path, img_name)
                self.patient_volumes[cls][patient_id].append(slice_img)

    def __len__(self):
        return sum([len(patients) for patients in self.patient_volumes.values()])

    def __getitem__(self, idx):
        all_patients = [(cls, patient_id) for cls, patients in self.patient_volumes.items() for patient_id in
                        patients.keys()]
        cls, patient_id = all_patients[idx]
        volume1 = self.patient_volumes[cls][patient_id]

        same_class = random.random() > 0.5
        if same_class:
            # different patient from the same class
            other_patients = [p for c, p in all_patients if c == cls and p != patient_id]
            other_patient_id = random.choice(other_patients)
            volume2 = self.patient_volumes[cls][other_patient_id]
        else:
            # different patient from the different class
            other_cls = 'NC' if cls == 'AD' else 'AD'
            other_patients = list(self.patient_volumes[other_cls].keys())
            other_patient_id = random.choice(other_patients)
            volume2 = self.patient_volumes[other_cls][other_patient_id]

        # Read slices for volume1 and convert to 3D tensor
        slices1 = [read_image(slice_path).float() for slice_path in volume1]
        if self.transform:
            slices1 = [self.transform(s) for s in slices1]
        volume1 = torch.stack(slices1, dim=0).squeeze()

        # Read slices for volume2 and convert to 3D tensor
        slices2 = [read_image(slice_path).float() for slice_path in volume2]
        if self.transform:
            slices2 = [self.transform(s) for s in slices2]
        volume2 = torch.stack(slices2, dim=0).squeeze()

        label = torch.tensor([0 if same_class else 1], dtype=torch.float32)

        return {"volume1": volume1,
                "volume2": volume2,
                "label": label}

""" 
from torch.utils.data import DataLoader
if __name__ == '__main__':
    dataloader = DataLoader(
        dataset=ContrastiveDataset(Config.TRAIN_DIR),
        shuffle=True,
        batch_size=3,
        num_workers=1,
        drop_last=True
    )
    for batch in dataloader:
        print("Batch:")
        print("volume1 shape:", batch["volume1"].shape)
        print("volume2 shape:", batch["volume2"].shape)
        print("label:", batch["label"])
        print()
        break
"""
