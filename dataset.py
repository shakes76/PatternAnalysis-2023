import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from PIL import Image

class ISICDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.image_dir, self.image_files[idx].replace('.jpg', '_superpixels.png'))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
# Set your paths
train_image_dir = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Training_Data'
valid_image_dir = 'C:\\Users\\25060\\Desktop\\ISIC-2017_Validation_Data'

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

train_dataset = ISICDataset(train_image_dir, transform=transform)
valid_dataset = ISICDataset(valid_image_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False)


