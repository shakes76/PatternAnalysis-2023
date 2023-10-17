from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

class customDataset(Dataset):
    def __init__(self, images, GT,transform):
        super(customDataset, self).__init__()

        self.images = images
        self.GT = GT

        self.transform = transform
        
        pass

    def __len__(self):
        if len(self.images) == len(self.GT):
            return(len(self.images))
        else:
            print(">>data and ground truth not same lenght")

    def __getitem__(self, idx):    
        img = Image.open(self.images[idx])
        GT = Image.open(self.GT[idx])

        if self.transform:
            img, GT = self.transform((img, GT))
            
            
        return img, GT
    

def data_sorter(img_root, gt_root):
    img_path = sorted(os.listdir(img_root))
    gt_path = sorted(os.listdir(gt_root))

    images_train = []
    gt_train = []
    images_test = []
    gt_test = []

    #create a list of indices in the data
    indices = list(range(len(img_path)))

    train_indices, test_indices = train_test_split(indices, train_size=0.8, test_size=0.2)

    for idx in train_indices:
        if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
            continue
        image = os.path.join(img_root, img_path[idx])
        gt = os.path.join(gt_root, gt_path[idx])

        images_train.append(image)
        gt_train.append(gt)

    for idx in test_indices:
        if img_path[idx] == 'ATTRIBUTION.txt' or img_path[idx] == 'LICENSE.txt':
            continue
        image = os.path.join(img_root, img_path[idx])
        gt = os.path.join(gt_root, gt_path[idx])

        images_test.append(image)
        gt_test.append(gt)

    return images_train, gt_train, images_test, gt_test