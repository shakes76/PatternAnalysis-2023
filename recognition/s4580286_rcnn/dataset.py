'''
Loads the data from the file directory and processes to output

'''
import torch 
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import torchvision.transforms as transforms 
from PIL import Image

class MoleData(Dataset): 
    def __init__(self, image_path, mask_path, diag_path, device, transform):
        '''
        Takes inputs of image_path, diagnostic_path(.csv) and mask_paths

        ''' 
        self.image_path = image_path
        self.mask_path = mask_path
        self.transform = transform
        self.diagnosis = pd.read_csv(diag_path)
    
    def __getitem__(self, index): 
        ''' 
        Load and process single sample
        Outputs : labels 
        
        '''
        image_idx = self.diagnosis.iloc[index]["image_id"]
        melanoma_status = self.diagnosis.iloc[index]["melanoma"]

        if melanoma_status: 
            label = 1 #Has melanoma 
        else: 
            label = 0 

        im_path = os.path.join(self.image_path, image_idx + ".jpg")
        image = Image.open(im_path)
        image = Image.convert('RGB') #Expecting channels 