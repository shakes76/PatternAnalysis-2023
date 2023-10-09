import torch
import os
from PIL import Image
from torchvision.transforms import Compose, Grayscale, ToTensor, Lambda
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class ImageDataset(Dataset):
    def __init__(self, directory, image_transforms=None):
        self.directory = directory
        self.image_files = sorted(os.listdir(directory))
        self.image_transforms = image_transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.image_files[index])
        image = Image.open(image_path).convert("L")
        
        if self.image_transforms:
            image = self.image_transforms(image)
        
        return image

def process_dataset(batch_size=8,
                    train_dir="/Users/Eiji/Documents/ML/Data/OASIS_Brain/keras_png_slices_train", 
                    test_dir="/Users/Eiji/Documents/ML/Data/OASIS_Brain/keras_png_slices_test", 
                    val_dir="/Users/Eiji/Documents/ML/Data/OASIS_Brain/keras_png_slices_validate"):
    
    # Given images are preprocessed with the size of 256 x 256
    image_transforms = Compose([
        Grayscale(),
        ToTensor(), 
        Lambda(lambda t: (t * 2) - 1),
    ])

    train_data = ImageDataset(directory=train_dir, image_transforms=image_transforms)
    test_data = ImageDataset(directory=test_dir, image_transforms=image_transforms)
    val_data = ImageDataset(directory=val_dir, image_transforms=image_transforms)

    # Combine all three datasets into single dataset for training
    combined_data = ConcatDataset([train_data, test_data, val_data])
    
    return DataLoader(combined_data, batch_size=batch_size, shuffle=True)
