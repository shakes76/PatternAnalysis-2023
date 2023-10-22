import pandas as pd
import os
import random
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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

def get_image_paths_from_directory(directory_path, valid_extensions=[".jpg", ".jpeg", ".png"]):
    """
    Get image paths from directory with valid extensions
    """
    if not os.path.exists(directory_path):
        raise ValueError(f"The provided directory {directory_path} does not exist.")

    all_images = []
    for image_file in os.listdir(directory_path):
        if any(image_file.endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(directory_path, image_file)
            all_images.append(image_path)
    return all_images

# Function to extract patient ID from image path
def extract_patient_id(image_path):
    """
    Extract patient ID from image path
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return base_name.split('_')[0]