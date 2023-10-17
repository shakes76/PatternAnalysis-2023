from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt
from torchvision.utils import make_grid

# Initialize constant variables for dataset creation and loading
TRAIN_DATASET_PATH = './AD_NC/train'
TEST_DATASET_PATH = './AD_NC/test'

BATCH_SIZE = 32

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),

])

# Define the initial dataset from the root directory without train/test specific transforms
full_dataset = ImageFolder(root=TRAIN_DATASET_PATH)

# Define the lengths of train and validation datasets
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_dataset.dataset.transform = transform
val_dataset.dataset.transform = transform
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

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)