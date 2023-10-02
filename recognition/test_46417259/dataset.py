import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms.v2 as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

# global variables
batch_size = 128
workers = 0

# local paths
train_path = '/Users/minhaosun/Documents/COMP3710_local/data/AD_NC/train'
test_path = '/Users/minhaosun/Documents/COMP3710_local/data/AD_NC/test'

# transforms
train_transforms = transforms.Compose([
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.ToTensor()
])

# load the trainset
trainset = dset.ImageFolder(root=train_path,
                            transform=train_transforms
                           )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

# load the testset
testset = dset.ImageFolder(root=test_path,
                           transform=test_transforms
                           )

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                        shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
print("Device: ", device)

print(f'trainset has classes {trainset.class_to_idx} and {len(trainset)} images')
print(f'testset has classes {testset.class_to_idx} and {len(testset)} images')

# Plot some training images
train_batch = next(iter(trainloader))
print(train_batch[0][0].shape)
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(train_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()

# the following data visualisation code is modified based on code at
# https://github.com/pytorch/tutorials/blob/main/beginner_source/basics/data_tutorial.py
# published under the BSD 3-Clause "New" or "Revised" License
# full text of the license can be found in this project at BSD_new.txt
figure = plt.figure(figsize=(8, 8))
cols, rows = 5, 5

labels_map = {0: 'AD', 1: 'NC'}

for i in range(1, cols * rows + 1):
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[train_batch[1][i].tolist()])
    plt.axis("off")
    plt.imshow(np.transpose(train_batch[0][i].squeeze(), (1,2,0)), cmap="gray")
plt.show()