import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

def get_dataloaders(batch_size, workers, image_size):
    # Create the dataset
    train_dataroot = "AD_NC/train"
    test_dataroot = "AD_NC/test"

    train_dataset = dset.ImageFolder(root=train_dataroot,
                                transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                            ]))

    test_dataset = dset.ImageFolder(root=test_dataroot,
                                transform=transforms.Compose([
                                transforms.Resize((image_size, image_size)),
                                transforms.ToTensor(),
                            ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=False, num_workers=workers)