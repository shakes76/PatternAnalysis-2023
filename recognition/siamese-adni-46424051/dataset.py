##################################   dataset.py   ##################################
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class DatasetTrain(Dataset):
    def __init__(self, path, transforms):
        super(DatasetTrain, self).__init__()
        self.transforms = transforms
        self.CN, self.AD = self.load_images(path)

    def load_images(self, path):
        CN = []
        AD = []
        
        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "CN")):
            filePath = os.path.join(path, "CN", filePath)
            CN.append(Image.open(filePath).convert("L"))

        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "AD")):
            filePath = os.path.join(path, "AD", filePath)
            AD.append(Image.open(filePath).convert("L"))

        return CN, AD
    
if __name__=="__main__":
    dataset = DatasetTrain(os.path.join(os.getcwd(), "images", "train"), None)