from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import Tuple
from torch import Tensor
from PIL import Image

# For training and testing the ADNI dataset for Alzheimer's disease was utilised which can be found here; https://adni.loni.usc.edu/
# Go to DOWNLOAD -> ImageCollections -> Advanced Search area to download the data

# Same size utilized from Google's paper on ViT
# Images are converted to this size x size
_size = 224

# Use the computed mean and std for normalization in transformations
# Please see utils for the method use for this calculation
_train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAdjustSharpness(sharpness_factor=0.9, p=0.1),
        transforms.Grayscale(num_output_channels=1),  # Converts to grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.14147302508354187,
            std=0.2420143187046051,
        ),
    ]
)

_test_transform = transforms.Compose(
    [
        transforms.Resize(_size),
        transforms.CenterCrop(_size),
        transforms.Grayscale(num_output_channels=1),  # Converts to grayscale
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.14147302508354187,
            std=0.2420143187046051,
        ),
    ]
)


def get_train_val_loaders() -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess data, split into training and validation, and create data loaders.

    Args:
        None

    Returns:
        tuple: A tuple containing the training, validation, and testing data loaders.
    """
    print("Loading training data...")
    train_dataset = datasets.ImageFolder(root="data/train", transform=_train_transform)
    print(f"training data loaded with {len(train_dataset)} samples.")

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=6)
    print(f"Training loader ready.\n")

    print("Loading validation data...")
    val_dataset = datasets.ImageFolder(root="data/val", transform=_test_transform)
    print(f"training data loaded with {len(val_dataset)} samples.")

    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=6)
    print(f"Validation loader ready.\n")

    print("Training and validation loaders ready.")

    return train_loader, val_loader


def get_test_loader() -> DataLoader:
    """
    Load and preprocess test data and create data loader.

    Args:
        None

    Returns:
        DataLoader: testing data loader.
    """
    print("Loading testing data...")
    test_dataset = datasets.ImageFolder(root="data/test", transform=_test_transform)
    print(f"Testing data loaded with {len(test_dataset)} samples.")

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=6)
    print(f"Testing loader ready.\n")

    return test_loader


def get_user_image(image_dir: str) -> Tensor:
    """
    Load and preprocess user image and create tensor.

    Args:
        image_dir (str): the directory to the user's image they wish to have the predictions on.

    Returns:
        Tensor: user image tensor.
    """
    print("Loading user image...")
    user_image = Image.open(image_dir)
    print(f"User image loaded.")

    user_image_tensor = _test_transform(user_image).unsqueeze(0)
    print(f"User loader ready.\n")

    return user_image_tensor


# Please note within the development environment the data was loaded in the following structure
# This structure will have to be replicated to make use of the ImageFolder function
# None of the folders contain any of the same data - was manually separated

# data/
#    ├── test/
#    │   ├── AD/
#    │   └── NC/
#    ├── train/
#    │   ├── AD/
#    │   └── NC/
#    └── val/
#        ├── AD/
#        └── NC/
