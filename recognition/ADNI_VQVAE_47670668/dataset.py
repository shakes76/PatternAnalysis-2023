from torchvision import transforms
from torchvision.datasets import ImageFolder

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Initialize constant variables for dataset creation and loading
TRAIN_DATASET_PATH = './AD_NC/train'
TEST_DATASET_PATH = './AD_NC/test'

BATCH_SIZE = 32

transform = transforms.Compose([
    # Convert images to tensor
    transforms.ToTensor(),

    # Rotates the image by a random angle between -15 to +15 degrees.
    transforms.RandomRotation(degrees=15),

    # With a probability of 50% (p=0.5), flips the image horizontally.
    transforms.RandomHorizontalFlip(p=0.5),

    # With a probability of 50% (p=0.5), flips the image vertically.
    transforms.RandomVerticalFlip(p=0.5),

    # Crops the image to a size of 224x224 pixels.
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),

    # Randomly changes the brightness, contrast, saturation, and hue of the image.
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
])

# Use image folder due to directory structure of dataset
train_dataset = ImageFolder(TRAIN_DATASET_PATH, transform=transform)
test_dataset = ImageFolder(TEST_DATASET_PATH, transform=transform)

# Visualize first 16 training images
def imshow(img_tensor):
    img = img_tensor.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# Get a batch of images
num_images_to_display = 16
images, labels = zip(*[train_dataset[i] for i in range(num_images_to_display)])

# Convert images to a grid and display
grid = make_grid(list(images), nrow=4)
imshow(grid)