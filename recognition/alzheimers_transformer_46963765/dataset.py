#import it
import torch
from torch.utils.data import DataLoader
import torchvision

'''
Builder class for the ADNI dataset
'''
class ADNI_Dataset:
    
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self._train_root = "./recognition/alzheimers_transformer_46963765/data/train"
        self._test_root = "./recognition/alzheimers_transformer_46963765/data/test"
    
    def get_train_loader(self, location=None, transform=False):
        if location != None:
            root_path = location
        else:
            root_path = self._train_root
            
        train_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

        
    def get_test_loader(self, location=None, transform=False):
        
        if location != None:
            root_path = location
        else:
            root_path = self._test_root
            
        test_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return test_loader 
    

dataset = ADNI_Dataset()
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()


