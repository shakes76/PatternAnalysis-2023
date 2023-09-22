import sys, os, time
import torch, torchvision
from torch.utils import data
from matplotlib import pyplot as plt
from traceback import format_exc


class Dataset(data.Dataset):
    """
    custom dataset

    """
    def __init__(self, train=True):

        self._dataset_path = "/home/groups/comp3710/"
        self._dataset_type = "train/" if train else "test/"

        self._dataset = "ADNI/"
        self._dataset_subdir = "AD_NC/"

        self._dataset_classes = {0: "AD/", 1: "NC/"}

        self._mean = (0.5, 0.5, 0.5)
        self._std_dev = (0.5, 0.5, 0.5)

        self._transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(64),
                torchvision.transforms.Normalize(self._mean, self._std_dev),
            ]
        )

        self._x = torchvision.datasets.ImageFolder(
            root=f"./debug/{self._dataset_type}",
            transform=self._transform,
        )

        print(self._x.imgs)


    def __len__(self):
        return len(self._x.imgs)
    
    
    def __getitem__(self, i):
        return self._x.loader(self._x.imgs[i][0]), self._x.imgs[i][1]
    

    def shape(self):
        return self._x.shape



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize(64),
                #torchvision.transforms.Normalize(mean, std_dev),
            ]
        )

    train_dataset = torchvision.datasets.ImageFolder(
            root="./debug/train/",
            transform=transform,
        )

    train_loader = data.DataLoader(
        dataset=train_dataset, batch_size=128, shuffle=False,
    )

    for images, labels in train_loader:

        plt.imshow(images[4].permute(1, 2, 0))
        plt.show()

        print(labels[4], flush=True)

        #print(images[0].shape)

        sys.exit()


if __name__ == "__main__":
    main()