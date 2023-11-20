"""
This file contains the pytorch_lightning dataloading class and functions for image patching
ADNI valadation split is done manually through file manipulation (80/20) split from train
"""
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip, RandomResizedCrop, Normalize
    
class ADNIDataModule(LightningDataModule):
    """ADNI data module in lighting.

    Args:
        batch_size: interger of Images in each batch
        image_size: List of image dimensions (height, width)
        num_workers: interger of workers for multi proccessing
                        Major issue use 0 in all cases
    """
    def __init__(
        self, 
        batch_size,
        image_size,
        num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        #set up root for image locations
        self.test_data_root = './Data/ADNI/AD_NC/test'
        self.train_data_root = './Data/ADNI/AD_NC/train'
        self.val_data_root = './Data/ADNI/AD_NC/val'

        #Setup transfroms
        #Transfrom for both test and validation sets
        self.test_transform = Compose([
                                ToTensor(), 
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        #transform for traing set
        self.train_transform = Compose([ 
                                RandomHorizontalFlip(),
                                RandomResizedCrop((image_size), scale=(0.8, 1.0), ratio=(0.9, 1.0)),
                                ToTensor(),
                                Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        #Set seed as to be reproducable
        pl.seed_everything(42)
        
        #Intiallising splits of data
        self.test = ImageFolder(root=self.test_data_root, transform=self.test_transform)
        self.train = ImageFolder(root=self.train_data_root, transform=self.train_transform)
        self.val = ImageFolder(root=self.val_data_root, transform=self.test_transform)

        #check imagefolders are working (if dir is correct)
        print("Image folders loaded")

    #Pytorch_lightning functions for loading split data
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=self.num_workers, shuffle=True)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, shuffle=False)

def img_to_patch(x, patch_size):
    """
    Args:
        x: Tensor representing the image of shape [B, C, H, W]
        patch_size: Number of pixels per dimension of the patches (integer)
    """
    B, C, H, W = x.shape
    x = x.reshape(B, C, H // patch_size, patch_size, W // patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)  # [B, H', W', C, p_H, p_W]
    x = x.flatten(1, 2)  # [B, H'*W', C, p_H, p_W]
    x = x.flatten(2, 4)  # [B, H'*W', C*p_H*p_W]
    return x

