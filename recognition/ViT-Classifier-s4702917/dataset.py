# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

# Parameters
batchSize = 64

# transforms
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = ImageFolder("/home/groups/comp3710/ADNI/AD_NC/train", transform=transform)
testset = ImageFolder("/home/groups/comp3710/ADNI/AD_NC/test", transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

# constant for classes
classes = ('AD', 'NC')
channels = 1

# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
	if one_channel:
		img = img.mean(dim=0)
	img = img / 2 + 0.5     # unnormalize
	npimg = img.numpy()
	if one_channel:
		plt.imshow(npimg, cmap="Greys")
	else:
		plt.imshow(np.transpose(npimg, (1, 2, 0)))