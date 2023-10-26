"""
This file contains the pytorch_lightning dataloading class and functions for image patching
lightingdatamodule class for both CIFAR10 (testing purposes)
                        and ADNI datasets
ADNI valadation split is done manually through file manipulation
"""
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torchvision.utils import make_grid
from torchvision.datasets import CIFAR10, ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, Resize, Normalize, CenterCrop


class CIFAR10DataModule(LightningDataModule):
    """Pytorch_lighning module that will handle loading data into each epoch
        Will also handle downloading and preproccessing of data"""
    def __init__(
        self, 
        batch_size,
        image_size,
        num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = './Data/CIFAR10'

        #Setup transfroms
        #Transfrom for both test and validation sets
        self.test_transform = Compose([ToTensor(), Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        #transform for traing set
        self.train_transform = Compose([ 
                                RandomHorizontalFlip(),
                                RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.0)),
                                ToTensor(),
                                Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        

        #Splitting trickery as val set shouldnt use train_transfrom
        #but need to keep test and val sperate
        pl.seed_everything(42)
        train_set = CIFAR10(root=self.data_root, train=True, transform=self.train_transform, download=True)
        self.train, _ = random_split(train_set, [45000, 5000])

        val_set = CIFAR10(root=self.data_root, train=True, transform=self.test_transform, download=True)
        _, self.val = random_split(val_set, [45000, 5000])

        self.test = CIFAR10(root=self.data_root, train=False, transform=self.test_transform, download=True)

    #Pytorch_lightning functions for loading split data
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=self.num_workers, shuffle=True)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, shuffle=False)
    
class ADNIDataModule(LightningDataModule):
    """ADNI data module in lighting.

    Args:
        batch_size: interger of Images in each batch
                    Can lower memory load (typically 64 if cpu, 128 for gpu)
        image_size: List of image dimensions (height, width)
        num_workers: interger of workers for multi proccessing
    """
    def __init__(
        self, 
        batch_size,
        image_size,
        num_workers=0):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
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

        print("Image folders loaded")

    #Pytorch_lightning functions for loading split data
    def train_dataloader(self):
        return DataLoader(self.train, self.batch_size, num_workers=self.num_workers, shuffle=True)    

    def test_dataloader(self):
        return DataLoader(self.test, self.batch_size, num_workers=self.num_workers, shuffle=False)
    
    def val_dataloader(self):
        return DataLoader(self.val, self.batch_size, num_workers=self.num_workers, shuffle=False)


    

#TODO Make img_to_patch use num_patches
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

