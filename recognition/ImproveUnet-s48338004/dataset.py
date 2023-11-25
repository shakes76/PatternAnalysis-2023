import os  # For interacting with the operating system.
import torch  # Import PyTorch for deep learning functionalities.

from torch.utils.data import Dataset  # Abstract class representing a dataset.
from PIL import Image  # Python Imaging Library for opening, manipulating, and saving images.
from torchvision import transforms  # Common image transformations.

# Determine if CUDA (GPU support) is available, use it, otherwise default to CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dataset: {torch.cuda.is_available()}")  # Display whether CUDA is available.


class ISICDataset(Dataset):
    """Define a custom dataset class from the torch.utils.data.Dataset class"""
    def __init__(self, image_dir, mask_dir=None, img_transform=None, mask_transform=None, img_size=None):
        # Initialization method for the dataset object.
        self.img_size = img_size  # Resize images to this size.
        self.image_dir = image_dir  # Directory containing the images.
        self.mask_dir = mask_dir  # Optional: directory containing the masks.
        self.images = os.listdir(image_dir)  # List of filenames in the image directory.
        self.img_transform = img_transform  # Transformations to apply to the images.
        self.mask_transform = mask_transform  # Transformations to apply to the masks.

    def __len__(self):
        """Return the total number of items in the dataset."""
        
        return len(self.images)

    def __getitem__(self, idx):
        """Retrieve image and mask."""
        img_path = os.path.join(self.image_dir, self.images[idx])  # Path to the image.
        image = Image.open(img_path).convert("RGB")  # Open and convert the image to RGB.
        image = image.resize(self.img_size, Image.BILINEAR)  # Resize the image if size is specified.

        mask = None  # We start by assuming there is no mask.
        if self.mask_dir:  # If a mask directory is provided.
            mask_name = self.images[idx].replace('.jpg', '_segmentation.png')  # Mask filename.
            mask_path = os.path.join(self.mask_dir, mask_name)  # Path to the mask.

            if os.path.exists(mask_path):  # Check if the mask file exists.
                mask = Image.open(mask_path).convert("L")  # Open and convert the mask to greyscale.
                mask = mask.resize(self.img_size, Image.NEAREST)  # Resize the mask if size is specified.
            else:
                print(f"File not found: {mask_path}")  # Log a message if the mask file is not found.

        # Apply transformations if specified.
        if self.img_transform:
            image = self.img_transform(image)  # Apply image transformations.

        if mask is not None and self.mask_transform:  # Only apply mask transform if mask is present.
            mask = self.mask_transform(mask)  # Apply mask transformations.

        return image, mask  # Return the processed image and mask.


def get_mask_transform():
    """Define and return the transformations to apply to the mask"""
    
    return transforms.Compose([
        transforms.ToTensor()  # Convert the mask to a PyTorch tensor.
    ])


def get_transform():
    """Define and return the transformations to apply to the images"""
    
    return transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
        # Normalize the tensor with the standard mean and standard deviation of the ImageNet dataset.
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
