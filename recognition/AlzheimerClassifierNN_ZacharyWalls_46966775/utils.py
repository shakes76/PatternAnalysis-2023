import matplotlib as plt
import torch
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
from collections import defaultdict
from PIL import Image


def display_images(images, labels, predictions):
    plt.figure(figsize=(10, 7))
    for i, image in enumerate(images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            image[0], cmap="gray"
        )  # Assuming grayscale images. If RGB, remove [0].
        plt.title(f"True: {labels[i]}, Predicted: {predictions[i]}")
        plt.axis("off")
    plt.show()


def display_feature_maps(model, image, layers_to_display):
    plt.figure(figsize=(20, 20))
    with torch.no_grad():
        for layer in layers_to_display:
            x = layer(image)
            # Assuming the feature maps are in shape (batch, channels, height, width)
            for i in range(x.size(1)):
                plt.subplot(8, 8, i + 1)  # assuming you have 64 filters
                plt.imshow(x[0][i].cpu().numpy(), cmap="gray")
                plt.axis("off")
    plt.show()


def backward_hook(module, grad_input, grad_output):
    print("Inside " + module.__class__.__name__ + " backward hook:")
    print("grad_input", grad_input)
    print("grad_output", grad_output)
    print("=" * 50)
    return grad_input


# This function was used for the normalization calculations of the dataset
def compute_mean_std(loader):
    """
    Compute the mean and standard deviation of the dataset for all RGB channels.
    """
    mean = torch.zeros(3)  # Initialize mean for each RGB channel
    squared_mean = torch.zeros(3)  # Initialize squared mean for each RGB channel
    for images, _ in loader:
        # Calculate mean and squared mean for each channel separately
        mean += images.mean(dim=(0, 2, 3))
        squared_mean += (images**2).mean(dim=(0, 2, 3))

    mean /= len(loader)
    squared_mean /= len(loader)

    # Calculate standard deviation for each channel
    std = (squared_mean - mean**2) ** 0.5
    return mean.tolist(), std.tolist()


# If you wish to transpose the data into 3D scans this can be applied for the data loading
class BrainScan3DDataset(Dataset):
    def __init__(self, root_dir, transform=None, label=None):
        self.root_dir = root_dir
        self.transform = transform
        self.patient_data = self._load_data()
        self.uuids = list(self.patient_data.keys())
        self.label = label

    def _load_data(self):
        patient_data = defaultdict(list)
        for filename in os.listdir(self.root_dir):
            if filename.endswith(".jpeg"):
                uuid = "_".join(filename.split("_")[:-1])
                patient_data[uuid].append(os.path.join(self.root_dir, filename))

        # Sort the data based on the image number
        for uuid, slices in patient_data.items():
            slices.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        return patient_data

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        patient_uuid = self.uuids[idx]
        image_paths = self.patient_data[patient_uuid]
        # Loading and transforming each slice
        slices = [
            self.transform(Image.open(p).convert("L")) for p in image_paths
        ]  # Convert to grayscale
        # Stacking along the depth dimension
        tensors = torch.stack(slices, dim=1)  # (channels, depth, height, width)
        label = torch.tensor(self.label, dtype=torch.long)
        return tensors, label
