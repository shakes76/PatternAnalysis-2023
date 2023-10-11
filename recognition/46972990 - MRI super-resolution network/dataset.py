import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# Paths to the images
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

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
    def __init__(self, path):
        self.resize_transform = transforms.Resize((IMG_HEIGHT, IMG_WIDTH))
        self.data = datasets.ImageFolder(root=path, transform=self.resize_transform)
        
        self.down_transform = transforms.Compose([
            DownsampleTransform(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
        ])
        self.orig_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        orig_img, _ = self.data[idx]
        return self.down_transform(orig_img), self.orig_transform(orig_img)

def get_train_loader():
    dataset = SuperResolutionDataset(TRAIN_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    return data_loader

def get_test_loader():
    dataset = SuperResolutionDataset(TEST_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader

# Get the train and test loaders
train_loader = get_train_loader()
test_loader = get_test_loader()

# Extract a single batch from the train loader
data_iter = iter(train_loader)
downscaled_images, original_images = next(data_iter)

# Convert the images from tensor format back to PIL for display
def tensor_to_PIL(tensor):
    tensor = (tensor + 1) / 2.0
    tensor = tensor.squeeze(0)
    return transforms.ToPILImage()(tensor)

# Plot the images (plot 5 to ensure it is working)
num_images_to_display = 5

for i in range(num_images_to_display):
    plt.figure(figsize=(10,5))
    
    # Display downscaled image
    plt.subplot(1, 2, 1)
    plt.imshow(tensor_to_PIL(downscaled_images[i]))
    plt.title("Downscaled Image")
    
    # Display original image
    plt.subplot(1, 2, 2)
    plt.imshow(tensor_to_PIL(original_images[i]))
    plt.title("Original Image")
    
    plt.show()