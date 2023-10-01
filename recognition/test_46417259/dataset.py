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
# print(type(train_batch))
# print(len(train_batch))
# print(type(train_batch[0]))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(train_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()