import torch, torchvision
from torch.utils import data
from matplotlib import pyplot as plt


try:
    open("./README.md")
    machine = "local"
except FileNotFoundError as err:
    machine = "rangpur"


class Dataset(data.Dataset):
    """
    custom dataset

    """
    def __init__(self, train=True):

        """ train or test dataset """
        self._dataset_type = "train/" if train else "test/"

        """ path to dataset on rangpur """
        self._dataset_path = "/home/groups/comp3710/"
        self._dataset = "ADNI/"
        self._dataset_subdir = "AD_NC/"

        """ default mean and std dev """
        self._mean = (0.5, 0.5, 0.5)
        self._std_dev = (0.5, 0.5, 0.5)

        """ default transforms: to pytorch tensor, normalise to mean and std dev """
        self._transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(self._mean, self._std_dev),
            ]
        )

        """ specify path to dataset """
        if machine == "local":
            path = f"./debug/{self._dataset_type}"
        elif machine == "rangpur":
            path = f"{self._dataset_path}{self._dataset}{self._dataset_subdir}{self._dataset_type}"

        """ load image data from specified path to dataset """
        self._dataset = torchvision.datasets.ImageFolder(
            root=path, transform=self._transform, target_transform=self.to_one_hot,
        )

        """ create a pytorch dataloader for the dataset """
        self._loader = data.DataLoader(
            dataset=self._dataset, batch_size=16, shuffle=True
        )

        """ display parameters associated with the dataset """
        print(f"{self._dataset_type}, {len(self._loader) = }, {self._dataset.class_to_idx = }")

    def loader(self):
        """
        method to get data loader
        
        """
        return self._loader
    
    def to_one_hot(self, target):
        """
        convert class encoding to one hot
        - initially implemented for the ViT
        - not used in the SR-GAN model

        """
        one_hot = [0] * len(self._dataset.class_to_idx)
        one_hot[target] = 1
        return torch.Tensor(one_hot)


def main():
    """
    main function: used to test the functionality of the dataloader module
    (not used for training the actual model)

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device, flush=True)

    """ train dataset """
    train_loader = Dataset(train=True).loader()

    plt.figure()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        t_min, t_max = torch.min(images[i]), torch.max(images[i])
        image = (images[i] - t_min) / (t_max - t_min)
        print(t_max - t_min)

        if i == 0: print(f"train: {images.shape = }, {labels.shape = }, {t_min = }, {t_max = }")
        if i >= 9: break
        
        plt.subplot(3, 3, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu())
        plt.title(labels[i], size=8)

    """ save training batch """
    if machine == "local":
        plt.savefig("./debug/train.png")
    elif machine == "rangpur":
        plt.savefig("./outputs/train.png")

    """ test dataset """
    test_loader = Dataset(train=False).loader()

    plt.figure()
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        t_min, t_max = torch.min(images[i]), torch.max(images[i])
        image = (images[i] - t_min) / (t_max - t_min)

        if i == 0: print(f"train: {images.shape = }, {labels.shape = }, {t_min = }, {t_max = }")
        if i >= 9: break

        plt.subplot(3, 3, i + 1)
        plt.imshow(image.permute(1, 2, 0).cpu())
        plt.title(labels[i], size=8)

    """ save testing batch """
    if machine == "local":
        plt.savefig("./debug/test.png")
    elif machine == "rangpur":
        plt.savefig("./outputs/test.png")
    

if __name__ == "__main__":
    main()