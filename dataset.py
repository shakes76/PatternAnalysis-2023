import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data
import os


def get_dataloaders(batch_size: int, workers: int, image_resize: int, dataroot: str, rgb: bool):
    """Returns a training dataloader, test dataloader, and validation dataloader.
    Assumes that dataroot contains 'train' and 'test' folders.
    """
    def get_patient_id_from_filename(filepath):
        filename = os.path.basename(filepath)
        return filename.split('_')[0]
    
    # Get file paths
    train_dataroot = os.path.join(dataroot, "train")
    test_dataroot = os.path.join(dataroot, "test")

    # Convert to grayscale if necessary
    if not rgb:
        transform = transforms.Compose([
                                    transforms.Resize((image_resize, image_resize)),
                                    transforms.ToTensor(),
                                    transforms.Grayscale()
                                ])
    else:
        transform = transforms.Compose([
                                    transforms.Resize((image_resize, image_resize)),
                                    transforms.ToTensor(),
                                ])
    
    # Initiate datasets and apply transformations
    full_train_dataset = dset.ImageFolder(root=train_dataroot, transform=transform)
    test_dataset = dset.ImageFolder(root=test_dataroot, transform=transform)

    approx_split_idx = int(0.8 * len(full_train_dataset))
    
    # Find next patient after 0.8 mark and split dataset there
    for idx in range(approx_split_idx, len(full_train_dataset)):
        current_patient_id = get_patient_id_from_filename(full_train_dataset.imgs[idx][0])
        next_patient_id = get_patient_id_from_filename(full_train_dataset.imgs[idx + 1][0])
        if current_patient_id != next_patient_id:
            split_idx = idx + 1
            break
    
    train_indices = list(range(split_idx))
    validation_indices = list(range(split_idx, len(full_train_dataset)))

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    validation_sampler = torch.utils.data.SubsetRandomSampler(validation_indices)

    # Create the dataloader for each dataset
    train_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size,
                                                num_workers=workers, sampler=train_sampler)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)

    validation_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=batch_size,
                                                    num_workers=workers, sampler=validation_sampler)
    
    return train_loader, test_dataloader, validation_loader