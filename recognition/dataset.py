from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os


class ISICDataset(Dataset):
    def __init__(self, image_dir, mask_dir=None, image_transform=None, mask_transform=None, img_size=(384, 512)):
        self.img_size = img_size
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_name).convert("RGB")
        image = image.resize(self.img_size, Image.BILINEAR)

        if self.mask_dir:
            mask_name = os.path.join(self.mask_dir, self.images[idx].replace('.jpg', '_segmentation.png'))
            mask = Image.open(mask_name).convert("L")  # Grayscale
            mask = mask.resize(self.img_size, Image.NEAREST)
        else:
            mask = None

        if self.image_transform:
            image = self.image_transform(image)
        if mask and self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask
    
