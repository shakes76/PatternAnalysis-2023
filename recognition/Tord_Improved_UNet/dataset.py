import os
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image


#dataset
class TrainingDataset(VisionDataset):
    def __init__(self, root, data_transform=None, target_transform=None, test=False):
        super(TrainingDataset, self).__init__(root, transforms=None, transform=data_transform, target_transform=target_transform)
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []

        if(os.path.exists(os.path.join(root, 'tensorsets/image.pt'))):
            if(test):
                self.images = torch.load(os.path.join(root, 'tensorsets/test_images.pt'))
                self.masks = torch.load(os.path.join(root, 'tensorsets/test_masks.pt'))
                return
            self.images = torch.load(os.path.join(root, 'tensorsets/images.pt'))
            self.masks = torch.load(os.path.join(root, 'tensorsets/masks.pt'))
            return
    
        data_folder = os.path.join(root, 'ISIC2018_Task1-2_Training_Input_x2')
        mask_folder = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth_x2')
        
        for i in range(10000):
            image_name = "ISIC_{:07d}.jpg".format(i)
            mask_name = image_name.replace('.jpg', '_segmentation.png')
            try:
                image = Image.open(os.path.join(data_folder, image_name))
                mask = Image.open(os.path.join(mask_folder, mask_name))
            except:
                continue
            image = self.data_transform(image)
            mask = self.target_transform(mask)
            self.images.append(image)
            self.masks.append(mask)
        torch.save(self.images, 'recognition/Tord_Improved_UNet/tensorsets/test_images.pt')
        torch.save(self.masks, 'recognition/Tord_Improved_UNet/tensorsets/test_masks.pt')
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        return image_path, mask_path

class CustomNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        # Normalize each modality independently
        for i in range(tensor.shape[0]):
            tensor[i] = (tensor[i] - self.mean[i]) / self.std[i]
        return tensor
    
mean = [0.7084, 0.5822, 0.5361]
std = [0.0948, 0.1099, 0.1240]

# Example usage:
data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    CustomNormalize(mean, std)
    
])

target_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    CustomNormalize(mean, std)
])

@staticmethod
def load_data():
    return TrainingDataset(root='recognition/Tord_Improved_UNet/data', data_transform=data_transform, target_transform=target_transform)

@staticmethod
def load_test():
    return TrainingDataset(root='recognition/Tord_Improved_UNet/data', data_transform=data_transform, target_transform=target_transform, test=True)

    

