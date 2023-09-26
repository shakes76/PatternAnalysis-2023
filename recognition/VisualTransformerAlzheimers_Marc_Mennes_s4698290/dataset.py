import os
from torchvision.io import read_image
import torch

#turns the ADNI raw data into a pytorch dataset to be used with a dataloader

class ADNI(torch.utils.data.Dataset):

    def __init__(self, path, transform=None, test=False):
        self.path = path #path not ending with a /
        self.transform = transform
        self.isTest = test

        #get a list of image names
        if not test:
            AD = os.listdir(self.path + "/AD_NC/train/AD")
            NC = os.listdir(self.path + "/AD_NC/train/NC")
        else:
            AD = os.listdir(self.path + "/AD_NC/test/AD")
            NC = os.listdir(self.path + "/AD_NC/test/NC")
        
        #map the images to indices for __getitem__
        self.data = AD + NC
        self.lastAD = len(AD) - 1

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        imageName = self.data[idx]
        ADLabel = 0

        #build the path to the image and find the label
        if idx <= self.lastAD:
            pathToImage = self.path + "/AD_NC" + ("/test" if self.isTest else "/train") + "/AD/"
            ADLabel = 1
        else:
            pathToImage = self.path + "/AD_NC" + ("/test" if self.isTest else "/train") + "/NC/"
        
        image = read_image(pathToImage + imageName)

        if self.transform is not None:
            image = self.transform(image)

        return image, ADLabel #ADLabel = 1 means alzheimer's, 0 means normal
    
#calculate norm and std of the dataset
def find_mean_std(path, transform = None):
    device = torch.device("cuda")
    dataset = ADNI(path, transform=transform)

    allImages = torch.cat([dataset.__getitem__(i)[0].to(device) for i in range(len(dataset))])

    print(allImages)
    print(allImages.size())

    return allImages.mean(dtype=torch.float32), allImages.float().std()
#cropped outputs for a 0-255 format image (tensor(41.0344, device='cuda:0'), tensor(64.2557, device='cuda:0'))


