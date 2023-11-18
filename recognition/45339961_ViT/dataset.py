""" Dataloader for loading and preprocessing the dataset. """

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_datasets(root_dir, 
                    train_transform, 
                    valid_transform, 
                    test_transform, 
                    datasplit):
    """ Create train, validation and test datasets.

    Args:
        root_dir (string): Directory with all the images.
        train_transform (Transform): Transform to be performed on training data
        valid_transform (Transform): Transform to be performed on validation data
        test_transform (Transform): Transform to be performed on test data
        datasplit (float): Split ratio for train and validation data

    Returns:
        Datasets: Train, validation and test datasets
    """

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

    # Create datasets
    train_data = ImageFolder(root=train_dir, transform=train_transform)
    valid_data = ImageFolder(root=train_dir, transform=valid_transform)

    # Overwrite samples
    train_data.samples = train_samples
    valid_data.samples = valid_samples

    return train_data, valid_data, test_data

def create_dataloaders(root_dir,
                        train_transform,
                        valid_transform, 
                        test_transform, 
                        batch_size, 
                        datasplit, 
                        num_workers=NUM_WORKERS):
    """ Create train, validation and test dataloaders.

    Args:
        root_dir (string): Directory with all the images.
        train_transform (Transform): Transform to perform on training data
        valid_transform (Transform): Transform to perform on validation data
        test_transform (Transform): Transform to perform on test data
        batch_size (int): Batch size for dataloaders
        datasplit (float): Split ratio for train and validation data

    Returns:
        Dataloaders: Train, validation and test dataloaders
    """

    # Get datasets
    train_data, valid_data, test_data = create_datasets(root_dir=root_dir,
                                                        valid_transform=valid_transform,
                                                        train_transform=train_transform,
                                                        test_transform=test_transform,
                                                        datasplit=datasplit)

    # Create dataloaders
    train_loader = DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
    
    valid_loader = DataLoader(valid_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)
    
    test_loader = DataLoader(test_data,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    return train_loader, valid_loader, test_loader