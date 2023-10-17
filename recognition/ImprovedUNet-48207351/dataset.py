import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


# Define the dataset root directory and transformation for data preprocessing.
data_root = '/ISIC-2017_Training_Data'

def transform_image():
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize the images to 128x128.
        transforms.ToTensor(),  # Convert to a PyTorch tensor.
    ])
    return transform

# Loading the data (source: https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/image_segmentation/semantic_segmentation_unet/dataset.py)
class ISICDataLoader(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_mask.gif"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


