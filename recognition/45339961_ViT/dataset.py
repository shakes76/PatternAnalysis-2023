""" Data loader for loading and preprocessing the dataset. """

# import os
# import torch
# import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
# import matplotlib.pyplot as plt
# from PIL import Image

class ADNIDataset(Dataset):
    """ ADNI dataset. """
    def __init__(self,
                root_dir,
                transform=None):
        self.root_dir = root_dir
        self.transform = transform





def basic_loader(dir, transform, shuffle=True, batch_size=64, num_workers=4):
    data = datasets.ImageFolder(root=dir, transform=transform)
    loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True)
    return loader



# if __name__ == "__main__":
#     BATCH_SIZE = 32

#     transform = transforms.Compose([
#         transforms.CenterCrop((192, 192)),
#         # transforms.RandomRotation(degrees=90),
#         transforms.ToTensor(),
#     ])

#     train_loader, test_loader = create_dataloaders("D:/AD_NC", transform=transform, batch_size=BATCH_SIZE)
#     print(train_loader.dataset.get_class_counts())
#     print(test_loader.dataset.get_class_counts())

#     images, labels, names = next(iter(train_loader))
#     # images = images.numpy()
#     print(len(images))
#     print(images[0].shape)
#     print(names[0])
#     plt.imshow(images[0][0], cmap='gray')
#     plt.axis("off")
#     plt.show()