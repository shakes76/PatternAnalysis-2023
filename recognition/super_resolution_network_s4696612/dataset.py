from PIL import Image
from torch.utils.data import Dataset
import os

class ImageDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.data = os.listdir(directory)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path = os.path.join(self.directory, self.data[index])
        label = Image.open(image_path)
        image = label.resize((64,60))
        if self.transform:
            label = self.transform(label)
            image = self.transform(image)
        return image, label