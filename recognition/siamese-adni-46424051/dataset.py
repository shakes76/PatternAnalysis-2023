##################################   dataset.py   ##################################
import torch
from torch.utils.data import Dataset
import os
from PIL import Image

class DatasetTrain(Dataset):
    def __init__(self, path, transforms):
        super(DatasetTrain, self).__init__()
        self.transforms = transforms
        self.NC, self.AD = self.load_images(path)

    def load_images(self, path):
        NC = []
        AD = []
        
        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "NC")):
            filePath = os.path.join(path, "NC", filePath)
            NC.append(Image.open(filePath).convert("L"))

        # Load from filepath
        for filePath in os.listdir(os.path.join(path, "AD")):
            filePath = os.path.join(path, "AD", filePath)
            AD.append(Image.open(filePath).convert("L"))

        return NC, AD
    
if __name__=="__main__":
    dataset = DatasetTrain(os.path.expanduser("~/../../groups/comp3710/ADNI/AD_NC/train"), None)
