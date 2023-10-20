"""
Name: dataset.py
Student: Ethan Pinto (s4642286)
Description: Creates the data loader for loading and preprocessing the ADNI Brain Data.
"""
import os
import numpy as np
import csv
import torch
import math
import torchvision
from torch.utils.data import DataLoader, TensorDataset, Dataset
# from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

class_labels = {
    'AD': 1,
    'NC': 0
}

train_data = []
test_data = []

train_dataroot = "C:/Users/Q/OneDrive/Desktop/COMP3710/REPORT/ADNI/AD_NC/train"
test_dataroot = "C:/Users/Q/OneDrive/Desktop/COMP3710/REPORT/ADNI/AD_NC/test"

# train_dataroot = "/home/groups/comp3710/ADNI/AD_NC/train"
# test_dataroot = "/home/groups/comp3710/ADNI/AD_NC/test"

# # Iterate through the subdirectories (AD and NC) and assign labels accordingly
# for class_name, label in [("AD", 1), ("NC", 0)]:
#     class_directory = os.path.join(train_dataroot, class_name)
#     for filename in os.listdir(class_directory):
#         if filename.endswith('.jpeg') or filename.endswith('.png'):
#             image_path = os.path.join(class_name, filename)
#             train_data.append((image_path, label))

#     class_directory = os.path.join(test_dataroot, class_name)
#     for filename in os.listdir(class_directory):
#         if filename.endswith('.jpeg') or filename.endswith('.png'):
#             image_path = os.path.join(class_name, filename)
#             test_data.append((image_path, label))

# # Specify the desired CSV file name
# csv_file = 'train_labelled.csv'

# # Write the collected data to a CSV file
# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['filename', 'label'])  # Write the header row
#     writer.writerows(train_data)  # Write the data

# csv_file = 'test_labelled.csv'

# with open(csv_file, 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerow(['filename', 'label'])  # Write the header row
#     writer.writerows(test_data)  # Write the data


class ADNIDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")  # Ensure the image is in RGB mode
        label = torch.tensor(int(self.img_labels.iloc[idx, 1]))

        if self.transform:
            image = self.transform(image)  # Apply the transformation to the image

        return image, label


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),  # Resize images to a common size
    torchvision.transforms.ToTensor(),  # Convert images to tensors
    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

siamese_train_dataset = ADNIDataset(annotations_file='train_labelled.csv', img_dir=train_dataroot, transform=transform)
mlp_train_dataset = ADNIDataset(annotations_file='train_labelled.csv', img_dir=train_dataroot, transform=transform)
test_dataset = ADNIDataset(annotations_file='test_labelled.csv', img_dir=test_dataroot, transform=transform)


def make_paired_datasets(dataset):
    X_pairs, y_pairs = [], []

    for i in range(len(dataset)):
        img_A, label_A = dataset[i]

        # Select a random image with a different class label
        while True:
            j = np.random.randint(len(dataset))
            img_B, label_B = dataset[j]
            if label_A != label_B:
                break

        X_pairs.append([img_A, img_B])
        y_pairs.append(1)  # Pair has the same class label, so label is 1

    # Add pairs with different class labels (label 0)
    for i in range(len(dataset)):
        img_A, label_A = dataset[i]
        j = np.random.randint(len(dataset))
        img_B, label_B = dataset[j]
        while label_A == label_B:
            j = np.random.randint(len(dataset))
            img_B, label_B = dataset[j]

        X_pairs.append([img_A, img_B])
        y_pairs.append(0)  # Pair has different class labels, so label is 0

    X_pairs = np.array(X_pairs)
    y_pairs = np.array(y_pairs)

    return X_pairs, y_pairs


# Define the train dataset for Siamese network
siamese_train_dataset = ADNIDataset(annotations_file='train_labelled.csv', img_dir=train_dataroot, transform=transform)

# Create paired datasets for Siamese network
siamese_X, siamese_y = make_paired_datasets(siamese_train_dataset)
# Convert numpy arrays to PyTorch tensors
siamese_X_tensor = torch.Tensor(siamese_X)
siamese_y_tensor = torch.Tensor(siamese_y)

# Create datasets from tensors
siamese_train_dataset = TensorDataset(siamese_X_tensor, siamese_y_tensor)


mlp_train_dataset = ADNIDataset(annotations_file='train_labelled.csv', img_dir=train_dataroot, transform=transform)
test_dataset = ADNIDataset(annotations_file='test_labelled.csv', img_dir=test_dataroot, transform=transform)


batch_size = 32

# Create data loaders
trainloader = DataLoader(siamese_train_dataset, batch_size=batch_size, shuffle=True)
mlp_loader = DataLoader(mlp_train_dataset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Get the size of each dataset
train_dataset_size = len(trainloader.dataset)
mlp_dataset_size = len(mlp_loader.dataset)
test_dataset_size = len(testloader.dataset)

print("Size of the Siamese Training Dataset:", train_dataset_size)
print("Size of the MLP Training Dataset:", mlp_dataset_size)
print("Size of the Test Dataset:", test_dataset_size)



# Iterate through a few batches from the Siamese loader
print("Siamese Loader:")
for batch_idx, (data, labels) in enumerate(trainloader):
    print(f"Batch {batch_idx + 1}:")
    print("Data shape:", data.shape)  # Print the shape of the data tensor
    print("Labels:", labels)  # Print the labels
    if batch_idx >= 2:  # Print the first 3 batches as an example
        break

# Iterate through a few batches from the MLP loader
print("\nMLP Loader:")
for batch_idx, (data, labels) in enumerate(mlp_loader):
    print(f"Batch {batch_idx + 1}:")
    print("Data shape:", data.shape)  # Print the shape of the data tensor
    print("Labels:", labels)  # Print the labels
    if batch_idx >= 2:  # Print the first 3 batches as an example
        break

# Iterate through a few batches from the test loader
print("\nTest Loader:")
for batch_idx, (data, labels) in enumerate(testloader):
    print(f"Batch {batch_idx + 1}:")
    print("Data shape:", data.shape)  # Print the shape of the data tensor
    print("Labels:", labels)  # Print the labels
    if batch_idx >= 2:  # Print the first 3 batches as an example
        break


# # Fetch a batch of data
# data_iter = iter(trainloader)
# images, labels = next(data_iter)

# # Display a few images from the dataset in two rows
# def show_images(images, labels, class_names):
#     num_images = len(images)
#     num_columns = int(math.ceil(num_images / 2))
#     fig, axes = plt.subplots(2, num_columns, figsize=(15, 7))

#     for i in range(2):
#         for j in range(num_columns):
#             index = i * num_columns + j
#             if index < num_images:
#                 image = images[index].permute(1, 2, 0)
#                 axes[i, j].imshow(image)
#                 axes[i, j].set_title(class_names[labels[index]])
#                 axes[i, j].axis('off')

#     plt.tight_layout()
#     plt.show()

# # Load and display the first batch of images with axis titles in two rows
# for images, labels in mlp_loader:
#     show_images(images, labels, ["NC","AD"])
#     break

