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
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        label = image
        result = image.resize((60, 64))
        return result, label