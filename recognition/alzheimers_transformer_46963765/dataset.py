#import it
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


'''
Builder class for the ADNI dataset
'''
class ADNI_Dataset:
    
    def __init__(self, batch_size=16):
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
    
    # image starts off at 3x240x256 need to convert to 1x240x240
    def get_transformation(self, type):
        if type == "train":
            transform_method = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(240),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.12385,), (0.2308,))
            #maybe add a random crop of decent size
            ])
        else:
            transform_method = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(240),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.12385,), (0.2308,))
            ])
            
        return transform_method
    
    
'''
Seperate class to visualise models in different ways
'''
class Model_Visualiser:
    def __init__(self, loader) -> None:
        self._loader = loader
    
    #method to visialise the images contianed by class
    def visualise(self):    
        displayed_count = 0
        for batch in self._loader:
            images, labels = batch 

            # Iterate through the images
            for image in images:
                # Display the image and its shape
                plt.imshow(image.permute(1, 2, 0))
                plt.title(f"Image Shape: {image.shape}")
                plt.axis('off')
                plt.show()

                displayed_count += 1

                # Max 10 images to be shown
                if displayed_count >= 10:
                    break
            # Max 10 images to be shown
            if displayed_count >= 10:
                break
            
    def getMeanAndStd(self):
        mean = 0
        std = 0
        samples = 0
        
        for batch in self._loader:
            images, labels = batch
            
            samples += 1
            mean += torch.mean(images, dim=[0,2,3])
            std +=  torch.mean(images**2, dim=[0,2,3])
            

        # Calculate the mean and std for each channel
        mean = mean / samples
        std = torch.sqrt(std / samples - mean ** 2)
        
        return mean, std

    



dataset = ADNI_Dataset()
train_loader = dataset.get_train_loader()
test_loader = dataset.get_test_loader()
visuals = Model_Visualiser(train_loader); visuals.visualise()

