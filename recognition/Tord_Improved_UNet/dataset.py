import os
import torch
from torchvision import transforms
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import torch.nn as nn

class TrainingDataset(VisionDataset):
    def __init__(self, root, data_transform=None, target_transform=None):
        super(TrainingDataset, self).__init__(root, transforms=None, transform=data_transform, target_transform=target_transform)
        self.data_transform = data_transform
        self.target_transform = target_transform
        self.images = []
        self.masks = []
        
        
        
        if(os.path.exists('recognition/Tord_Improved_UNet/tensorsets/images.pt')):
            self.images = torch.load('recognition/Tord_Improved_UNet/tensorsets/images.pt')
            self.masks = torch.load('recognition/Tord_Improved_UNet/tensorsets/masks.pt')
            return
        
        data_folder = os.path.join(root, 'ISIC2018_Task1-2_Training_Input_x2')
        mask_folder = os.path.join(root, 'ISIC2018_Task1_Training_GroundTruth_x2')
        i = 0
        for image_name in os.listdir(data_folder):
            i+=1
            mask_name = image_name.replace('.jpg', '_segmentation.png')  # Assumes naming conventions for masks
            image = Image.open(os.path.join(data_folder, image_name))
            mask = Image.open(os.path.join(mask_folder, mask_name))
            image = self.data_transform(image)
            mask = self.target_transform(mask)
            self.images.append(image)
            self.masks.append(mask)
        torch.save(self.images, 'recognition/Tord_Improved_UNet/tensorsets/images.pt')
        torch.save(self.masks, 'recognition/Tord_Improved_UNet/tensorsets/masks.pt')
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        mask_path = self.masks[idx]

        return image_path, mask_path

# Example usage:
data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

@staticmethod
def load():
    return TrainingDataset(root='recognition/Tord_Improved_UNet/data', data_transform=data_transform, target_transform=target_transform)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        prediction_sum = prediction.sum()
        target_sum = target.sum()
        dice = (2.0 * intersection + self.smooth) / (prediction_sum + target_sum + self.smooth)
        dice_loss = 1 - dice
        return dice_loss