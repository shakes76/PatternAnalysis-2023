# imports
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
from torchvision.datasets import ImageFolder

# Parameters
batchSize = 32

# transforms
class SquarePad:
	def __call__(self, image: torch.Tensor):
		_, h, w = image.shape
		max_wh = np.max([h, w])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')

train_transform = transforms.Compose(
	[transforms.ToTensor(),
	SquarePad(),
	# the images are grayscale already, with all channels equal,
	# this just converts it to single-channel.
	transforms.Grayscale(),
	transforms.RandomRotation(180, expand=True), # Completely random rotation.
	transforms.RandomResizedCrop(size=(256, 256), scale=(0.7,1), antialias=True), # a bit of data augmentation
	transforms.Normalize((0.5,), (0.5,))]
 )

# No data augmentation when testing.
test_transform = transforms.Compose(
	[transforms.ToTensor(),
	SquarePad(),
	# the images are grayscale already, with all channels equal,
	# this just converts it to single-channel.
	transforms.Grayscale(),
	transforms.Normalize((0.5,), (0.5,))]
 )

# datasets
trainset = ImageFolder("./data/ADNI/train", transform=train_transform)
validset = ImageFolder("./data/ADNI/validation", transform=test_transform)
testset = ImageFolder("./data/ADNI//test", transform=test_transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)

validloader = torch.utils.data.DataLoader(validset, batch_size=batchSize,
                                          shuffle=True, num_workers=0)


testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize,
                                         shuffle=False, num_workers=0)

# constant for classes
classes = ('AD', 'NC')
channels = 1