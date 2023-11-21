import torch
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import random
from torch.utils.data import Dataset, DataLoader
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

def extract_patient_id(image_path):
    """
    Extract the patient ID from image path.
    """

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return base_name.split('_')[0]

def load_data(train_images_paths_AD, train_images_paths_NC):
    """
    Load and split image dataset into training and validation sets whilst ensuring no patient overlap between sets
    """
    # Get image paths for training and test datasets

    all_train_images_paths_NC = get_image_paths_from_directory(train_images_paths_NC)
    all_train_images_paths_AD = get_image_paths_from_directory(train_images_paths_AD)

    # Extract unique patient IDs for training and test sets
    all_patient_ids_AD = list(set(extract_patient_id(path) for path in all_train_images_paths_AD))
    all_patient_ids_NC = list(set(extract_patient_id(path) for path in all_train_images_paths_NC))
    # Split patient IDs into training and validation sets (e.g., 80%, 20% split)
    train_patient_ids_AD, val_patient_ids_AD = train_test_split(all_patient_ids_AD, test_size=0.20, random_state=42)
    train_patient_ids_NC, val_patient_ids_NC = train_test_split(all_patient_ids_NC, test_size=0.20, random_state=42)
    # Map patient IDs back to image paths for training and validation sets
    train_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in train_patient_ids_AD]
    val_images_AD = [path for path in all_train_images_paths_AD if extract_patient_id(path) in val_patient_ids_AD]
    train_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in train_patient_ids_NC]
    val_images_NC = [path for path in all_train_images_paths_NC if extract_patient_id(path) in val_patient_ids_NC]


    return train_images_AD, train_images_NC, val_images_AD, val_images_NC

def create_data_loaders(train_images_AD, train_images_NC, val_images_AD, val_images_NC, batch_size):
    """
    Create data loaders for training and validation sets with specified transformations.
    """
    # Define the data transformation for train, validation, and test
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = ADNC_Dataset(train_images_AD, train_images_NC, transform=data_transforms['train'])
    val_dataset = ADNC_Dataset(val_images_AD, val_images_NC, transform=data_transforms['val'])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_dataloader, val_dataloader

def load_test_data(test_images_paths_AD, test_images_paths_NC):
    """
    Loads test data from specified directory and filters patient ID
    """
    all_test_images_paths_NC = get_image_paths_from_directory(test_images_paths_NC)
    all_test_images_paths_AD = get_image_paths_from_directory(test_images_paths_AD)

    all_patient_ids_AD_test = list(set(extract_patient_id(path) for path in all_test_images_paths_AD))
    all_patient_ids_NC_test = list(set(extract_patient_id(path) for path in all_test_images_paths_NC))

    # Map patient IDs back to image paths for test set
    test_images_AD = [path for path in all_test_images_paths_AD if extract_patient_id(path) in all_patient_ids_AD_test]
    test_images_NC = [path for path in all_test_images_paths_NC if extract_patient_id(path) in all_patient_ids_NC_test]

    return test_images_AD, test_images_NC
