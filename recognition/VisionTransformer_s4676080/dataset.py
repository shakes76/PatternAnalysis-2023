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