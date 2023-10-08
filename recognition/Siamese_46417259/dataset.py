import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms.v2 as transforms
# import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import random

# global variables
batch_size = 128
workers = 0

# file paths
train_path = '/Users/minhaosun/Documents/COMP3710_local/data/AD_NC/train'
test_path = '/Users/minhaosun/Documents/COMP3710_local/data/AD_NC/test'

# transforms
train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(240),
])

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(240),
])


class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, image_folder:dset.ImageFolder, show_debug_info:bool, random_seed=None) -> None:
        super().__init__()
        self.image_folder = image_folder
        self.image_folder_size = len(self.image_folder)
        self.debug_mode = show_debug_info

        # for reproducibility
        if random_seed is not None:
            random.seed(random_seed)
            torch.random.seed(random_seed)

    def __len__(self) -> int:
        return self.image_folder_size

    def __getitem__(self, index: int):
        """
        returns [img1, img2, similarity] or [img1, img2, similarity, filepath1, filepath2]
        where:
            img1 and img2 are tensor representations of images
            similarity is 1 if the two images are of the same class and 0 otherwise
            filepath1 and filepath2 are strings representing the last 15 characters
                of the images' filepaths excluding the .jpeg extension
        """
        img1, label1 = self.image_folder[index]
        similarity = random.randint(0, 1)

        match_found = False
        while not match_found:
            choice = random.randint(0, self.image_folder_size - 1)
            # make sure we do not pair the same image against itself
            if choice == index:
                continue

            img2, label2 = self.image_folder[choice]
            if similarity == 1:
                match_found = label1 == label2
            else:
                match_found = label1 != label2

        # only include the filepaths in the dataset if specifically requested
        if self.debug_mode:
            filepath1 = self.image_folder.imgs[index][0]
            filepath2 = self.image_folder.imgs[choice][0]
            return img1, img2, similarity, filepath1[-20:-5], filepath2[-20:-5]
        # otherwise save some memory
        return img1, img2, similarity
    
    def showing_debug_info(self) -> bool:
        return self.debug_mode


def load_data(training:bool, Siamese:bool, random_seed=None) -> torch.utils.data.DataLoader:
    if training:
        path = train_path
        transforms = train_transforms
    else:
        path = test_path
        transforms = test_transforms

    if random_seed is not None:
        random.seed(random_seed)
        torch.random.seed(random_seed)

    source = dset.ImageFolder(root=path, transform=transforms)
    print(f'dataset has classes {source.class_to_idx} and {len(source)} images')

    if Siamese:
        # loading paired data for the Siamese neural net
        # each data point is of format [img1, img2, similarity]
        # where similarity is 1 if the two images are of the same class and 0 otherwise
        dataset = PairedDataset(source, show_debug_info=False, random_seed=random_seed)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    else:
        # loading unitary training data for the MLP
        # each data point is of format [img, label]
        # where label is 1 if the image is of class AD and 0 otherwise
        loader = torch.utils.data.DataLoader(source, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
    return loader


#
# basic tests
#
def test_load_data_basic():
    dataloader = load_data(training=True, Siamese=True)
    next_batch = next(iter(dataloader))
    print(next_batch[0][0].shape)

def test_visualise_data_MLP():
    # Plot some training images
    dataloader = load_data(training=True, Siamese=False)
    next_batch = next(iter(dataloader))
    print(next_batch[0][0].shape)

    # the following data visualisation code is modified based on code at
    # https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/data_tutorial.py
    # published under the BSD 3-Clause "New" or "Revised" License
    # full text of the license can be found in this project at BSD_new.txt
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    labels_map = {0: 'AD', 1: 'NC'}

    for i in range(1, cols * rows + 1):
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[next_batch[1][i].tolist()])
        plt.axis("off")
        plt.imshow(np.transpose(next_batch[0][i].squeeze(), (1,2,0)), cmap="gray")
    plt.show()

def test_visualise_data_Siamese():
    # Plot some training images
    dataloader = load_data(training=True, Siamese=True)
    dataloader.dataset.debug_mode = True

    next_batch = next(iter(dataloader))
    print(next_batch[0][0].shape)
    print(next_batch[0][1].shape)

    cols, rows = 3, 3
    fig, axs = plt.subplots(rows, cols * 2)
    labels_map = {0: 'diff', 1: 'same'}

    for i in range(rows):
        for j in range(cols):
            axs[i,j*2].imshow(np.transpose(next_batch[0][i*rows+j].squeeze(), (1,2,0)), cmap="gray")
            axs[i,j*2+1].imshow(np.transpose(next_batch[1][i*rows+j].squeeze(), (1,2,0)), cmap="gray")
            axs[i,j*2].set_title(f"""{labels_map[next_batch[2][i*rows+j].tolist()]}, {next_batch[3][i*rows+j]}""")
            axs[i,j*2+1].set_title(next_batch[4][i*rows+j])
            axs[i,j*2].axis("off")
            axs[i,j*2+1].axis("off")
    plt.show()

def test_paired_dataset():
    source = dset.ImageFolder(root=train_path,
                                transform=train_transforms
                            )
    test = PairedDataset(source, show_debug_info=True)
    print(len(test))
    print(len(source))
    visualise_paired_data(test)

def visualise_paired_data(dataset: PairedDataset):
    if not dataset.showing_debug_info():
        raise NotImplementedError("PairedDataset must be initialised with show_debug_info=True")

    testloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
        
    next_batch = next(iter(testloader))
    print(next_batch[0][0].shape)
    print(next_batch[0][1].shape)

    cols, rows = 3, 3
    fig, axs = plt.subplots(rows, cols * 2)
    labels_map = {0: 'diff', 1: 'same'}

    for i in range(rows):
        for j in range(cols):
            axs[i,j*2].imshow(np.transpose(next_batch[0][i*rows+j].squeeze(), (1,2,0)), cmap="gray")
            axs[i,j*2+1].imshow(np.transpose(next_batch[1][i*rows+j].squeeze(), (1,2,0)), cmap="gray")
            axs[i,j*2].set_title(f"""{labels_map[next_batch[2][i*rows+j].tolist()]}, {next_batch[3][i*rows+j]}""")
            axs[i,j*2+1].set_title(next_batch[4][i*rows+j])
            axs[i,j*2].axis("off")
            axs[i,j*2+1].axis("off")
    plt.show()


if __name__ == "__main__":
    # Decide which device we want to run on
    device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
    print("Device: ", device)

    # test_load_data_basic()
    test_visualise_data_MLP()
    test_visualise_data_Siamese()
    # test_paired_dataset()


#
# deprecated code below
#
def load_train_Siamese() -> torch.utils.data.DataLoader:

    train_source = dset.ImageFolder(root=train_path, transform=train_transforms)

    trainset = PairedDataset(train_source, show_debug_info=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
    return trainloader

def load_test_Siamese() -> torch.utils.data.DataLoader:
    # loading paired testing data for the Siamese neural net
    # testloader is of format [img1, img2, similarity]
    test_source = dset.ImageFolder(root=test_path, transform=test_transforms)

    testset = PairedDataset(test_source, show_debug_info=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
    return testloader


def load_train() -> torch.utils.data.DataLoader:
    # loading unitary training data for the MLP
    train_source = dset.ImageFolder(root=train_path, transform=train_transforms)

def load_test() -> torch.utils.data.DataLoader:
    # load the testset
    testset = dset.ImageFolder(root=test_path,
                            transform=test_transforms
                            )
    print(f'testset has classes {testset.class_to_idx} and {len(testset)} images')

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers)
    
    return testloader
