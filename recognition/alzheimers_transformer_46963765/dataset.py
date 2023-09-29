#import it
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

'''
Builder class for the ADNI dataset
'''
class ADNI_Dataset:
    
    def __init__(self, batch_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self._train_root = "./recognition/alzheimers_transformer_46963765/data/train"
        self._test_root = "./recognition/alzheimers_transformer_46963765/data/test"
    
    def get_train_loader(self, location=None, transform=None):
        if location != None:
            root_path = location
        else:
            root_path = self._train_root
        
        if transform == None:
            transform = self.get_transformation("train")
            
        train_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        return train_loader

        
    def get_test_loader(self, location=None, transform=None):
        
        if location != None:
            root_path = location
        else:
            root_path = self._test_root
            
        if transform == None:
            transform = self.get_transformation("test")
            
        test_dataset = torchvision.datasets.ImageFolder(root=root_path, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True)
        return test_loader 
    
    def get_transformation(self, type):
        if type == "train":
            transform_method = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ])
        else:
            transform_method = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            
        return transform_method

dataset = ADNI_Dataset()
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()


