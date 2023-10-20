from torchvision import transforms
from torch.utils.data import DataLoader , Dataset
from torchvision import datasets
from pathlib import Path
from PIL import Image
# Create custom dataset for your data
from torchvision.io import read_image

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, target_size=(256, 256)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = list(image_dir.glob("*.jpg"))
        self.masks = list(mask_dir.glob("*.png")) if mask_dir else []  # Handle cases with no masks

        self.target_size = target_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = str(self.images[idx].absolute())

       
        image = read_image(image_path)
        if self.mask_dir:
            
            mask_path = str(self.masks[idx].absolute())
            mask = read_image(mask_path)
        
            # Resize images to the target size   
            resize_transform = transforms.Resize(self.target_size, interpolation= Image.NEAREST)
            image = resize_transform(image)
            mask = resize_transform(mask)
            

            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)

            return image/ 255, mask/ 255
        return image/ 255 , [0]
