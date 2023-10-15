import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def get_image_paths_from_directory(directory_path, valid_extensions=[".jpg", ".jpeg", ".png"]):
    """Returns list of all image paths with valid extensions in the provided directory."""
    if not os.path.exists(directory_path):
        raise ValueError(f"The provided directory {directory_path} does not exist.")
    
    all_images = []
    for image_file in os.listdir(directory_path):
        if any(image_file.endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, image_file)
            all_images.append(image_path)
    return all_images

train_images_paths_AD = '/Users/noammendelson/Documents/REPORT-COMP3710/AD_NC/train/AD'
train_images_paths_NC = '/Users/noammendelson/Documents/REPORT-COMP3710/AD_NC/train/NC'

all_train_images_paths_NC = get_image_paths_from_directory(train_images_paths_NC)
all_train_images_paths_AD = get_image_paths_from_directory(train_images_paths_AD)


def extract_patient_id(image_path):
    """Extract patient ID from image path."""
    base_name = os.path.splitext(os.path.basename(image_path))[0] # # Extract the base filename without the extension
    return base_name.split('_')[0] # Split by underscore and return the patient ID 

## Extract unique patient IDs
all_patient_ids_AD = list(set(extract_patient_id(path) for path in all_train_images_paths_AD))
all_patient_ids_NC = list(set(extract_patient_id(path) for path in all_train_images_paths_NC))

# Split patient IDs into training and validation sets (20%, 80% split)
train_patient_ids_AD, val_patient_ids_AD = train_test_split(all_patient_ids_AD, test_size=0.2, random_state=42) 
train_patient_ids_NC, val_patient_ids_NC = train_test_split(all_patient_ids_NC, test_size=0.2, random_state=42)

# Map patient IDs back to image paths
train_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in train_patient_ids_AD]
val_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in val_patient_ids_AD]
train_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in train_patient_ids_NC]
val_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in val_patient_ids_NC]


class ADNC_Dataset(Dataset):
        """
    A custom Dataset class for loading AD and NC images.
    
    Attributes:
    - image_paths: A list of paths to the image files.
    - transforms: Optional transformations to apply to the images.
    """
        def __init__(self, AD_image_paths, NC_image_paths, transform=None):
            self.AD_image_paths = AD_image_paths
            self.NC_image_paths = NC_image_paths

            # Creating a DataFrame
            AD_df = pd.DataFrame({
                'image_path': self.AD_image_paths,
                'label': [1]*len(self.AD_image_paths)  # 1 for AD
            })
            
            NC_df = pd.DataFrame({
                'image_path': self.NC_image_paths,
                'label': [0]*len(self.NC_image_paths)  # 0 for NC
            })

            self.data = pd.concat([AD_df, NC_df], axis=0).reset_index(drop=True)
            #test code
            # pd.set_option('display.max_colwidth', None)
            # print(self.data.head())
            self.transform = transform
            
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
             """Returns an image and its label (either 0 or 1)."""
             row = self.data.iloc[idx]
             image_path = row['image_path']
             label = torch.tensor(row['label'])
            
             # Open the image and convert to RGB
             image = Image.open(image_path).convert("RGB")
                
             if self.transform:
                 image = self.transform(image)
                
             return image, label
        
# Data normalisation and augmentation for training
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(240),  # Using 240 as the crop size- divisible common patch sizes  (i.e. 16x16)
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(240),  #Center cropping to 240x240
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create training and validation datasets
train_dataset = ADNC_Dataset(train_images_AD, train_images_NC, transform=data_transforms['train'])
val_dataset = ADNC_Dataset(val_images_AD, val_images_NC, transform=data_transforms['val'])

# Create training and validation dataloaders
dataloaders_dict = {
    'train': DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4),
    'val': DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
}