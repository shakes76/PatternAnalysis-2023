import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip, RandomResizedCrop, ColorJitter

import os
from PIL import Image
from torchvision.transforms.functional import to_pil_image


class ADNIDataset(Dataset):
  """ADNI dataset"""

  def __init__(self, directories, transform=None):
        self.image_paths = []
        self.labels = []

        # Assign a label to each directory (0 for AD, 1 for NC)
        for i, directory in enumerate(directories):
          for filename in os.listdir(directory):
              if filename.endswith('.jpeg'):
                  self.image_paths.append(os.path.join(directory, filename))
                  self.labels.append(i)

        self.transform = transform

  def __len__(self):
      return len(self.image_paths)

  def __getitem__(self, idx):
      image = Image.open(self.image_paths[idx])
      label = self.labels[idx]

      if self.transform:
        image = self.transform(image)  

      return image, label
  
def preprocess_and_save(original_directories, preprocessed_directories):
    for orig_dir, prep_dir in zip(original_directories, preprocessed_directories):
        if not os.path.exists(prep_dir):
            os.makedirs(prep_dir)
        for filename in os.listdir(orig_dir):
            if filename.endswith('.jpeg'):
                # Load the image
                image_path = os.path.join(orig_dir, filename)
                image = Image.open(image_path)
                
                # Apply transformations
                transformed_image = transform(image)
                
                # Save the transformed image
                save_path = os.path.join(prep_dir, filename)
                to_pil_image(transformed_image).save(save_path)

  

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
])


train_dataset_paths = ['/content/drive/MyDrive/dataset/AD_NC/train/AD',
                       '/content/drive/MyDrive/dataset/AD_NC/train/NC'
                    ]
test_dataset_paths = ['/content/drive/MyDrive/dataset/AD_NC/test/AD',
                       '/content/drive/MyDrive/dataset/AD_NC/test/NC'
                    ]


# Define directories to save preprocessed images
preprocessed_train_paths = [
    '/content/drive/MyDrive/dataset/AD_NC_preprocessed/train/AD',
    '/content/drive/MyDrive/dataset/AD_NC_preprocessed/train/NC'
]

preprocessed_test_paths = [
    '/content/drive/MyDrive/dataset/AD_NC_preprocessed/test/AD',
    '/content/drive/MyDrive/dataset/AD_NC_preprocessed/test/NC'
]

# Ran once to create the directory and images
# preprocess_and_save(train_dataset_paths, preprocessed_train_paths)
# preprocess_and_save(test_dataset_paths, preprocessed_test_paths)


transform = transforms.Compose([
        transforms.ToTensor()
])
train_dataset = ADNIDataset(train_dataset_paths, transform=transform)
test_dataset = ADNIDataset(test_dataset_paths, transform=transform)