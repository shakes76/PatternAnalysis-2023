# dataset.py

import torch
from torchvision import datasets, transforms

# Constants
BATCH_SIZE = 32
DATASET_PATH = './OASIS'
NUM_WORKERS = 1

# Data preprocessing
preproc_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize for grayscale
])

# Data loaders
train_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/train_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/validat_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)

test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(DATASET_PATH + '/test_images', transform=preproc_transform),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)