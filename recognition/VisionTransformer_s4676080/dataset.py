# This is the dataset file

#Importing all the required libraries
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class AlzheimerDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_AD=0, num_NC=0):
        self.root_dir = root_dir
        self.transform = transform

        self.AD_files = [os.path.join(root_dir, "AD", f) for f in os.listdir(os.path.join(root_dir, "AD")) if os.path.isfile(os.path.join(root_dir, "AD", f))]
        self.NC_files = [os.path.join(root_dir, "NC", f) for f in os.listdir(os.path.join(root_dir, "NC")) if os.path.isfile(os.path.join(root_dir, "NC", f))]
               
        if not self.AD_files:
            raise ValueError("No AD images found!")
        if not self.NC_files:
            raise ValueError("No NC images found!")
       
        self.AD_files = self.AD_files[:num_AD]
        self.NC_files = self.NC_files[:num_NC]        
        self.all_files = self.AD_files + self.NC_files

    def __len__(self):
        return len(self.all_files)
    
    def __getitem__(self, idx):
        image_path = self.all_files[idx]
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)                
        label = 1 if os.path.basename(os.path.dirname(image_path)) == 'AD' else 0
        return image, label