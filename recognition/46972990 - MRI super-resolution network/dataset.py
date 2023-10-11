import os
from modules import SuperResolutionDataset
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

# Paths to the images
BASE_PATH = "C:\\Users\\User\\OneDrive\\Bachelor of Computer Science\\Semester 6 2023\\COMP3710\\ADNI_AD_NC_2D\\AD_NC"
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

BATCH_SIZE = 32

def get_train_and_validation_loaders():
    full_dataset = SuperResolutionDataset(TRAIN_PATH)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size) #80% train and 20% validation
    val_size = total_size - train_size

    torch.manual_seed(42)
    
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, validation_loader

def get_test_loader():
    dataset = SuperResolutionDataset(TEST_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return data_loader

# Get the train and test loaders
train_loader, validation_loader = get_train_and_validation_loaders()
test_loader = get_test_loader()

# Extract a single batch from the train loader
data_iter = iter(train_loader)
downscaled_images, original_images = next(data_iter)

print(original_images[0].shape)

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