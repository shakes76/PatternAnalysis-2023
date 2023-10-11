from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch.nn as nn

# Image size and batch size
IMG_WIDTH = 256
IMG_HEIGHT = 240
BATCH_SIZE = 32

class DownsampleTransform:
    """
    This class will downsample an image by a default scale factor of 4.
    """
    def __init__(self, scale_factor=4):
        self.down_transform = transforms.Compose([
            transforms.Resize((IMG_HEIGHT // scale_factor, IMG_WIDTH // scale_factor))
        ])
    
    def __call__(self, img):
        return self.down_transform(img)
    
class SuperResolutionDataset(Dataset):
    """
    This class will create datasets consisting of downsampled images and original images.
    """
    def __init__(self, path):
        self.resize_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        ])
        self.data = datasets.ImageFolder(root=path, transform=self.resize_transform)
        
        self.down_transform = transforms.Compose([
            DownsampleTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.orig_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig_img, _ = self.data[idx]
        return self.down_transform(orig_img), self.orig_transform(orig_img)
    
class ESPCN(nn.Module):
    """
    Implementation adapted to PyTorch from Tensorflow from https://keras.io/examples/vision/super_resolution_sub_pixel/
    """
    def __init__(self, upscale_factor=4, channels=1):
        super(ESPCN, self).__init__()
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=5, padding=2, padding_mode='reflect')
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv2d(32, channels * (upscale_factor ** 2), kernel_size=3, padding=1, padding_mode='reflect')
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x