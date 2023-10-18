import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.utils.data
import random
from glob import glob
from torchvision.transforms import Resize
import torchvision.transforms as T




TRAINDATA = "ISIC/ISIC-2017_Training_Data/ISIC-2017_Training_Data"
TESTDATA = "ISIC/ISIC-2017_Test_v2_Data/ISIC-2017_Test_v2_Data"
VALIDDATA = "ISIC/ISIC-2017_Validation_Data/ISIC-2017_Validation_Data"
TRAINTRUTH = "ISIC/ISIC-2017_Training_Part1_GroundTruth/ISIC-2017_Training_Part1_GroundTruth"
TESTTRUTH = "ISIC/ISIC-2017_Test_v2_Part1_GroundTruth/ISIC-2017_Test_v2_Part1_GroundTruth"
VALIDTRUTH = "ISIC/ISIC-2017_Validation_Part1_GroundTruth/ISIC-2017_Validation_Part1_GroundTruth"

NUM_EPOCHS = 5
BATCH_SIZE = 4
WORKERS = 4

    
class RandomChoice(torch.nn.Module):
    def __init__(self, transforms):
       super().__init__()
       self.transforms = transforms

    def __call__(self, imgs):
        t = random.choice(self.transforms)
        return [t(img) for img in imgs]
    



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms
        self.transform1 = RandomChoice([
            T.RandomHorizontalFlip(), 
            T.RandomVerticalFlip()
        ])


    def __call__(self, image, mask):
        image = self.transforms(image)
        mask = self.transforms(mask)
        '''
        t = T.RandomRotation(degrees=360)
        state = torch.get_rng_state()
        image = t(image)
        torch.set_rng_state(state)
        mask = t(mask)
        '''
        return image, mask
    
    
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.custom_transform = Compose(transform)
        
        self.resize = Resize((224, 224))
        self.resize_mask = Resize((224, 224),Image.NEAREST)

        ignored = {"ISIC-2017_Test_v2_Data_metadata.csv","ISIC-2017_Training_Data_metadata.csv"}
        self.images = [x for x in os.listdir(image_dir) if x not in ignored]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_segmentation.png"))
        image = self.resize(Image.open(img_path))
        mask = self.resize_mask(Image.open(mask_path))

        image = np.array(image)
        mask = np.array(mask)
        mask[mask != 255.0] = 0.0
        #mask[mask == 255.0] = 1.0
        #print("dataset1",mask.max())
        #print(mask)
        #print(mask.mean())
        #print(mask.max())
        #print(mask.min())
        #mask = mask/255
        #mask = mask/255
        #mask.astype(int)

        
        if self.transform is not None:
            image, mask = self.custom_transform(image, mask)
            #print("dataset2",mask.max())

            #image = augmentations["image"]
            #mask = augmentations["mask"]
        sample = {'image': image, 'mask': mask}
        return sample

