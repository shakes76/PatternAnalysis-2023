import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        classes = os.listdir(data_dir)
        for class_id, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            image_files = os.listdir(class_path)
            self.image_paths.extend([os.path.join(class_path, img) for img in image_files])
            self.labels.extend([class_id] * len(image_files))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


data_dir = './recognition/48240983_ADNI/AD_NC/train'


transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
])


custom_dataset = CustomDataset(data_dir, transform=transform)


batch_size = 16
data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

for images, labels in data_loader:
   
    pass
