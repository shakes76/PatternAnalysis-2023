""" Data loader for loading and preprocessing the dataset. """

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# def basic_loader(dir, transform, shuffle=True, batch_size=64, num_workers=4):
#     data = ImageFolder(root=dir, transform=transform)
#     loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
#     return loader

def create_datasets(root_dir, train_transform, test_transform, datasplit):
    train_dir = root_dir + "/train"
    test_dir = root_dir + "/test"

    train_valid_data = ImageFolder(root=train_dir, transform=train_transform)
    test_data = ImageFolder(root=test_dir, transform=test_transform)

        # Extract patient IDs
    patient_ids = list(set([os.path.basename(path).split('_')[0] for path, _ in train_valid_data.samples]))

    # Shuffle and split the patient IDs
    num_train = int(datasplit * len(patient_ids))
    train_patient_ids = set(patient_ids[:num_train])
    valid_patient_ids = set(patient_ids[num_train:])

    # Split dataset based on patient IDs
    train_samples = [(path, label) for path, label in train_valid_data.samples if os.path.basename(path).split('_')[0] in train_patient_ids]
    valid_samples = [(path, label) for path, label in train_valid_data.samples if os.path.basename(path).split('_')[0] in valid_patient_ids]

    train_data = ImageFolder(root=train_dir, transform=train_transform)
    valid_data = ImageFolder(root=train_dir, transform=test_transform)

    train_data.samples = train_samples
    valid_data.samples = valid_samples

    return train_data, valid_data, test_data

def create_dataloaders(root_dir, train_transform, test_transform, batch_size, datasplit):
    train_data, valid_data, test_data = create_datasets(root_dir=root_dir,
                                                        train_transform=train_transform,
                                                        test_transform=test_transform,
                                                        datasplit=datasplit)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader, test_loader

# if __name__ == "__main__":
#     BATCH_SIZE = 32
#     IMAGE_WIDTH = 192

#     train_transform = transforms.Compose([
#         transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#     ])

#     test_transform = transforms.Compose([
#         transforms.Resize((IMAGE_WIDTH, IMAGE_WIDTH)),
#         transforms.Grayscale(num_output_channels=1),
#         transforms.ToTensor(),
#     ])

#     root_dir = "C:/Users/Jacqu/Downloads/AD_NC"

#     train_data, valid_data, test_data = create_datasets(root_dir=root_dir, 
#                                                         train_transform=train_transform,
#                                                         test_transform=test_transform,
#                                                         datasplit=0.8)
#     print(f"Number of classes: {len(train_data.classes)}")
#     print(f"Class names: {train_data.classes}")

#     data = train_data

#     import random
#     # Randomly select an image and display its details
#     # random_index = random.randint(0, len(train_data) - 1)
#     random_index = 0
#     img_tensor, label = data[random_index]
#     class_name = data.classes[label]

#     # Convert tensor back to PIL Image for visualization
#     to_pil = transforms.ToPILImage()
#     img_pil = to_pil(img_tensor)
#     img_size = img_tensor.shape

#     print(f"Random Image Details:")
#     print(f"Label: {class_name}")
#     print(f"Size: {img_size}")
#     img_path = data.samples[random_index][0]  # Get the path
#     filename = os.path.basename(img_path)
#     print(f"Filename: {filename}")
#     img_pil.show()

#     from collections import Counter

#     # Count occurrences of each label in train_data
#     label_counts = Counter([label for _, label in data.samples])

#     # Print number of images for each class
#     for class_idx, count in label_counts.items():
#         class_name = data.classes[class_idx]
#         print(f"Class: {class_name}, Number of Images: {count}")