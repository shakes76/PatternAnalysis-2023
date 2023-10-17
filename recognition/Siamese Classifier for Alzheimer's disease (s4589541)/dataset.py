# loading and preprocessing of data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils import show_image, train_valid_split

BATCH_SIZE = 4

# setup the transforms for the images
transform = transforms.Compose([
    # transforms.ToTensor()
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
    data = train_set[0]
    print(data)
    print(len(train_set))
    show_image(data)

    data = valid_set[0]
    print(data)
    print(len(valid_set))
    show_image(data)

    # print(data)
    # show_image(data)

    # data_iterator = iter(train_loader)
    # data = next(data_iterator)
    # print(data)
    # show_image(data)
