import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt

# Define dataset paths
TRAIN_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_train"  # Training data
TEST_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_test"   # Test data
VALIDATION_DATA_PATH = "/Users/yashmittal/Downloads/Pattern Recognition Project 02/keras_png_slices_data/keras_png_slices_seg_validate"  # Validation data
CHANNELS_IMG = 3

# Customized ImageDataset to read image data
class CustomImageData(Dataset):
    def __init__(self, data_dirs, transform=None):
        self.transform = transform
        self.image_files = []
        for dir_path in data_dirs:
            self.image_files += [os.path.join(dir_path, fname) for fname in os.listdir(dir_path)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image

def get_data_loader(image_size):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            [0.5 for _ in range(CHANNELS_IMG)],
            [0.5 for _ in range(CHANNELS_IMG)],
        )
    ])

    dataset = CustomImageData(data_dirs=[TRAIN_DATA_PATH, TEST_DATA_PATH, VALIDATION_DATA_PATH], transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader, dataset

def display_sample_images():
    loader, _ = get_data_loader(256)
    images = next(iter(loader))
    _, ax = plt.subplots(8, 8, figsize=(8, 8))
    plt.suptitle('Original images')
    index = 0
    for k in range(8):
        for kk in range(8):
            ax[k][kk].imshow((images[index].permute(1, 2, 0) + 1) / 2)
            index += 1

    if not os.path.exists("output_images"):
        os.makedirs("output_images")

    save_path = os.path.join("output_images", "initial_image.png")
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    display_sample_images()
