# loading and preprocessing of data
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import show_image, train_valid_split

BATCH_SIZE = 64

class OneChannel:
    """Custom transform class to discard extra image channels."""
    @staticmethod
    def __call__(img_tensor):
        return img_tensor[0]


# setup the transforms for the images
transform = transforms.Compose([
    transforms.Resize((256, 240)),
    transforms.ToTensor(),
    OneChannel()
])

# set up the datasets
train_set = datasets.ImageFolder(root="data/train", transform=transform)
valid_set = datasets.ImageFolder(root="data/valid", transform=transform)
test_set = datasets.ImageFolder(root="data/test", transform=transform)

# set up the dataloaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)



if __name__ == '__main__':
    # train_valid_split()
    # data = train_set[0]
    # print(data)
    # print(len(train_set))
    # show_image(data)
    # transformed = transforms.ToTensor(data)
    # print(transformed)

    data = valid_set[0]
    print(data)
    print(len(valid_set))
    print(data[0].size())
    # for channel in data[0]:
        # print(channel)
        # print(len(channel))
        # print(torch.max(channel))
    # show_image(data)
    # transformed = transforms.ToTensor()(data[0])
    # print(transformed)

    # print(data)
    # show_image(data)

    # data_iterator = iter(train_loader)
    # data = next(data_iterator)
    # print(data)
    # show_image(data)
