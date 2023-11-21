"""
dataset.py: Defines a custom dataset class for AD and NC image data, 
along with associated transformations and dataloaders for training and testing.
"""

#Importing all the required libraries
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class AlzheimerDataset(Dataset):
    """
    Class to load AD and NC image data.
    
    Parameters:
    - root_dir: Path to the directory containing AD and NC image folders.
    - transform: Image transformations to be applied.
    - num_AD: Limit on the number of AD images to use from the dataset.
    - num_NC: Limit on the number of NC images to use from the dataset.
    """
    
    def __init__(self, root_dir, transform=None, num_AD=0, num_NC=0):
        self.root_dir = root_dir
        self.transform = transform

        # Get the list of AD image files
        self.AD_files = [os.path.join(root_dir, "AD", f) for f in os.listdir(os.path.join(root_dir, "AD")) if os.path.isfile(os.path.join(root_dir, "AD", f))]
        
        # Get the list of NC image files
        self.NC_files = [os.path.join(root_dir, "NC", f) for f in os.listdir(os.path.join(root_dir, "NC")) if os.path.isfile(os.path.join(root_dir, "NC", f))]     
        
        if not self.AD_files:
            raise ValueError("No AD images found!")
        if not self.NC_files:
            raise ValueError("No NC images found!")
       
        # Limit the number of AD and NC images if specified
        self.AD_files = self.AD_files[:num_AD]
        self.NC_files = self.NC_files[:num_NC]   
        self.all_files = self.AD_files + self.NC_files

    def __len__(self):
        """
        Returns:
        - The total number of images in the dataset.
        """
        return len(self.all_files)    

    def __getitem__(self, idx):
        """
        Fetches an image and its label by index.
        
        Parameters:
        - idx: The index of the image to fetch.
        
        Returns:
        - image: The image tensor.
        - label: The label of the image (1 for AD, 0 for NC).
        """
        image_path = self.all_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)                
        label = 1 if os.path.basename(os.path.dirname(image_path)) == 'AD' else 0
        return image, label

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Just normalization for validation and test
test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# Creating datasets and dataloaders
train_dataset = AlzheimerDataset(root_dir='/content/drive/My Drive/full_dataset/ADNI_AD_NC_2D/AD_NC/train', transform=train_transforms, num_AD=10400, num_NC=11120)
test_dataset = AlzheimerDataset(root_dir='/content/drive/My Drive/full_dataset/ADNI_AD_NC_2D/AD_NC/test', transform=test_transforms, num_AD=2000, num_NC=2000)

# Define dataloaders to load data in batches
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Displaying counts
#print(f"Training set: AD={len(train_dataset.AD_files)} images, NC={len(train_dataset.NC_files)} images")
#print(f"Test set: AD={len(test_dataset.AD_files)} images, NC={len(test_dataset.NC_files)} images")
    
